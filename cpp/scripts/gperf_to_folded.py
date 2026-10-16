"""
Convert a gperftools CPU profile (CPUPROFILE=...) into "folded" stacks for
FlameGraph's flamegraph.pl, using addr2line for symbols. Written because
`perf` is blocked on the cluster's compute nodes (perf_event_paranoid = 4)
and the conda gperftools package ships without the pprof viewer.

Usage:
  CPUPROFILE=out.prof CPUPROFILE_FREQUENCY=1000 \
      LD_PRELOAD=$CONDA_PREFIX/lib/libprofiler.so build-prof/main_euroc ...
  python scripts/gperf_to_folded.py out.prof > out.folded
  flamegraph.pl out.folded > flame.svg
"""
import bisect
import collections
import os
import re
import shutil
import struct
import subprocess
import sys


def read_profile(path):
    data = open(path, 'rb').read()
    words = lambda off, n: struct.unpack_from(f'<{n}Q', data, off)
    hdr = words(0, 5)   # 0, 3, version 0, period_us, padding
    if hdr[0] != 0 or hdr[1] != 3:
        sys.exit('not a 64-bit gperftools CPU profile')
    off = 5 * 8
    stacks = collections.Counter()
    while True:
        count, depth = words(off, 2)
        off += 16
        pcs = words(off, depth)
        off += 8 * depth
        if count == 0 and depth == 1 and pcs[0] == 0:   # trailer
            break
        stacks[pcs] += count
    maps = data[off:].decode(errors='replace')
    return stacks, maps


def parse_maps(text):
    regions = []   # (start, end, file_offset, path)
    for line in text.splitlines():
        m = re.match(r'([0-9a-f]+)-([0-9a-f]+) (\S+) ([0-9a-f]+) \S+ \d+\s+(\S.*)?', line)
        if not m or 'x' not in m.group(3) or not m.group(5) or not m.group(5).startswith('/'):
            continue
        regions.append((int(m.group(1), 16), int(m.group(2), 16), int(m.group(4), 16),
                        m.group(5).strip()))
    regions.sort()
    return regions


def symbolize(addrs_by_file):
    addr2line = shutil.which('addr2line') or shutil.which('x86_64-conda-linux-gnu-addr2line')
    if addr2line is None:
        sys.exit('addr2line not found (binutils); activate the vio_cpp env')
    names = {}
    for path, addrs in addrs_by_file.items():
        addrs = sorted(addrs)
        if not os.path.exists(path):
            continue
        out = subprocess.run([addr2line, '-a', '-f', '-C', '-i', '-e', path] +
                             [hex(a) for a in addrs], capture_output=True, text=True).stdout
        cur, frames = None, []
        lines = out.splitlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith('0x'):
                if cur is not None:
                    names[(path, cur)] = frames
                cur, frames = int(lines[i], 16), []
                i += 1
            else:
                frames.append(lines[i])   # function name; next line is file:line
                i += 2
        if cur is not None:
            names[(path, cur)] = frames
    return names


def simplify(name):
    name = re.sub(r'\(.*\)( const)?$', '', name)                     # drop arguments
    name = re.sub(r'Eigen::internal::', 'Eigen::', name)
    if len(name) > 90:
        name = name[:87] + '...'
    return name.replace(';', ':')


def main():
    stacks, maps = read_profile(sys.argv[1])
    regions = parse_maps(maps)
    starts = [r[0] for r in regions]

    def locate(pc):
        i = bisect.bisect_right(starts, pc) - 1
        if i >= 0 and regions[i][0] <= pc < regions[i][1]:
            start, _, file_off, path = regions[i]
            return path, pc - start + file_off
        return None, pc

    addrs_by_file = collections.defaultdict(set)
    located = {}
    for pcs in stacks:
        for k, pc in enumerate(pcs):
            key = pc if k == 0 else pc - 1   # return address -> call site
            if key not in located:
                located[key] = locate(key)
                if located[key][0]:
                    addrs_by_file[located[key][0]].add(located[key][1])
    names = symbolize(addrs_by_file)

    folded = collections.Counter()
    for pcs, count in stacks.items():
        frames = []
        for k, pc in enumerate(pcs):
            path, addr = located[pc if k == 0 else pc - 1]
            syms = names.get((path, addr))
            if not syms or syms == ['??']:
                lib = os.path.basename(path) if path else '?'
                frames.append(f'[{lib}]')
            else:
                # addr2line -i lists the innermost inlined function first.
                frames.extend(simplify(s) for s in syms)
        frames.reverse()   # root first
        # Collapse runs of unknown frames in the same library.
        collapsed = [f for i, f in enumerate(frames) if i == 0 or f != frames[i - 1]]
        folded[';'.join(collapsed)] += count
    for stack, count in folded.most_common():
        print(f'{stack} {count}')


if __name__ == '__main__':
    main()
