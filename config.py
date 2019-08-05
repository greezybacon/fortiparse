from __future__ import print_function

import argparse
from collections import OrderedDict as dict
import csv
import random
import re
import shlex
import sys
import warnings

# ---- Config parsing --------------------------
class Config(str):
    "Config is a marker to indicate that `config` should be used in the output"
    "rather than `edit`"
    pass

def read_unset(start_line, config):
    _, var = start_line.split(' ', 1)
    return var, None

def read_set(start_line, config):
    global keywords
    _, var, value = start_line.split(' ', 2)
    while (value.count('"') - value.count('\\"')) % 2 == 1:
        line = next(config)
        if line.startswith('set') or line.startswith('end'):
            raise SyntaxError("Corrupt config?: [%d] %s" % (config.line_number, line))
        value += "\n" + line
    return var, tuple(x.strip('"') for x in shlex.split(value))

def read_subsection(start_line, config):
    _, name = start_line.lstrip().split(' ', 1)
    name = name.strip('"')
    def settings():
        for line in config:
            if line.startswith('config'):
                yield read_section(line, config)
            elif line.startswith('set'):
                yield read_set(line, config)
            elif line.startswith('unset'):
                yield read_unset(line, config)
            elif line.startswith('next'):
                break
            elif line.startswith('end'):
                warnings.warn("Potential corruption: Subsection `%s` starting "
                    "with `edit` ends with `end`" % (name,))
                # Add an extra `end` due to inconsistency in VDOM configs ...
                config.pushback('end')
                break
            else:
                raise SyntaxError("Corrupt config?: [%d] %s" % (config.line_number, line))
    return name, dict(settings())

def read_section(start_line, config):
    _, name = start_line.split(' ', 1)
    name = name
    def subsections():
        for line in config:
            if line.startswith('edit'):
                yield read_subsection(line, config)
            elif line.startswith('config'):
                yield read_section(line, config)
            elif line.startswith('set'):
                yield read_set(line, config)
            elif line.startswith('end'):
                break
    return Config(name), dict(subsections())

def iter_sections(config):
    pragma = []
    for line in config:
        if line.startswith('config'):
            yield read_section(line, config)
        elif line.startswith('#'):
            pragma.append(line)
        else:
            raise SyntaxError("Corrupt config?: [line %d] %s" % (config.line_number, line))
    yield '#pragma', pragma

class StrippedConfig(object):
    "StrippedConfig strips leading and trailing whitespace and keeps track of "
    "line numbers for syntax errors. It also allows for pushback during iteration "
    "for syntax corrections."
    def __init__(self, file):
        self.file = file
        self.line_number = 0
        self.buffer = []
        
    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.line_number += 1
            line = self.buffer.pop(0) if len(self.buffer) else next(self.file)
            line = line.lstrip().rstrip("\n")
            if line:
                return line
    next = __next__
    
    def pushback(self, line):
        self.buffer.append(line)

def merging_dict(tuples):
    """Extension to dict() which handles repeating keys"""
    rv = dict()
    for k, v in tuples:
        if k in rv:
            rv[k].update(v)
        else:
            rv[k] = v
    return rv

def parse_config(file):
    return merging_dict(iter_sections(StrippedConfig(file)))

# ---- Output ----------------

def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

def escape(x):
    return x.replace('\\', '\\\\').replace('"', '\\"')

def quote(x, force=False, char='"'):
    if x == '':
        return char + char
    elif force or ' ' in x or '\\' in x or '*' in x:
        return '%s%s%s' % (char, escape(x), char)
    return x
 
def print_pragma(section, output=sys.stdout):
    if '#pragma' in section:
        for line in section.pop('#pragma'):
            output.write("%s\n" % (line,))

def pretty_print_vdom_config(section, prefix="", output=sys.stdout):
    # Output vdom stubs
    output.write("\n%sconfig vdom\n" % (prefix,))
    for vdom in section['vdom']:
        output.write("%sedit %s\n%snext\n" % (prefix, vdom, prefix))
    output.write("%send\n" % (prefix,))

    # Output global, then each vdom
    output.write("\n%sconfig global\n" % (prefix,))
    pretty_print_config(section['global'], prefix, output)
    output.write("%send\n" % (prefix,))

    for vdom in section['vdom']:
    	# Emit the EDIT line here to compenstate for lack of indent in
    	# Fortinet config
        output.write("\n%sconfig vdom\n%sedit %s\n" % (prefix, prefix, vdom))
        pretty_print_config(section['vdom'][vdom])
        output.write("%send\n" % (prefix,))
    
def pretty_print_config(section, prefix="", output=sys.stdout):
    print_pragma(section)
    if 'vdom' in section and 'global' in section:
        pretty_print_vdom_config(section, prefix, output)
    else:
        pretty_print_section(section, prefix, output)

def pretty_print_section(section, prefix="", output=sys.stdout):
    for name, value in section.items():
        if type(value) is dict:
            if isinstance(name, Config):
                output.write("%sconfig %s\n" % (prefix, name))
                pretty_print_section(value, prefix + '    ', output)
                output.write("%send\n" % (prefix,))
            else:
                output.write("%sedit %s\n" % (prefix, quote(name,
                    force=not is_number(name))))
                pretty_print_section(value, prefix + '    ', output)
                output.write("%snext\n" % (prefix,))
        elif value is None:
            output.write("%sunset %s\n" % (prefix, quote(name)))
        else:
            output.write("%sset %s %s\n" % (prefix, name, " ".join(
                quote(x) for x in value)))

def print_csv(section, output=sys.stdout):
    headers = ['id']
    for v in section.values():
        for k in v.keys():
            if k not in headers:
                headers.append(k)
    writer = csv.DictWriter(output, headers)
    writer.writeheader()
    for id, props in section.items():
        row = props.copy()
        row.update({'id': (id,)})
        writer.writerow({k: "; ".join(x for x in v) if v else None for k, v in row.items()})

# ---- Config Merging ------------------------

def merge_section_left(left, right, keyprop=None):
    # If there's nothing to compare for merging, then just return the RHS
    if not keyprop or left is None:
        return right

    # Go through right items, add them to left
    for rsection, rprops in right.items():
        rkeyval = rprops.get(keyprop)
        for lsection, lprops in left.items():
            if rkeyval is not None and lprops.get(keyprop) == rkeyval:
                # TODO: Merge properties
                del left[lsection]
                if type(rprops) is dict:
                    left[rsection] = merge_section_left(lprops, rprops)
                else:
                    left[rsection] = rprops
                break
        else:
            # It's new, add it to the left
            if rsection in left:
                # It (somehow?) conflicts with a section on the left.
                warnings.warn("Merge conflict: %s already in left, but `%s` does not match `%s`"
                    % (rsection, rkeyval, keyprop))
            else:
                left[rsection] = rprops
    return left

def merge_section(left, right, name):
    return merge_section_left(left.get(name), right.get(name), 'uuid')

def translate_interfaces(config, ifmap):
    # Recurse for DICT, replace for TUPLE, and passthrough otherwise
    return dict((
        ifmap[k] if k in ifmap else k,
        translate_interfaces(v, ifmap)
            if type(v) is dict
            else tuple(ifmap[p] if p in ifmap else p for p in v)
                if type(v) is tuple
                else v
        ) for k, v in config.items()
    )

# ---- Pre-flight checks ----------------------
ipv4addr = re.compile(r'\b((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|\b)){4}\b')
def scrub_ipset(props):
    # TODO: Handle IP/mask (used in `router policy`)
    if len(props) > 1:
        return (sanitize_ip(props[0], props[1]), *props[1:])
    else:
        value = props[0]
        for m in ipv4addr.finditer(props[0]):
            value = value[:m.start()] + sanitize_ip(m.group(0)) + value[m.end():]
        return (value,)

snetworks = {}
def sanitize_ip(ip, mask='255.255.255.0'):
    iip = ip_number(ip)
    imask = ip_number(mask)
    if imask == 4294967295:
        # Search the snetworks table for the network
        imask2 = imask
        for m in range(1, 24):
            imask2 = (imask2 << 1) & 0xffffffff
            if iip & imask2 in snetworks:
                imask = imask2
                break
        else:
            # assume 24-bit
            imask = 0xffffff00
    network = iip & imask
    host = iip & ~imask
    if network not in snetworks:
        snetworks[network] = (10 << 24) + (random.randint(1<<8, (1<<24) - 1) & imask)
    return ip_repr(snetworks[network] + host)

def ip_number(ip, mask='255.255.255.255'):
    net = 0
    for i, m in zip(ip.split('.'), mask.split('.')):
        net = (net << 8) + (int(i) & int(m))
    return net

def ip_repr(ip):
    return '.'.join([str((ip >> i) & 255) for i in range(24, -1, -8)])

def sanitize(config, pwds=True, ips=True):
    for section, props in config.items():
        if type(props) is dict:
            sanitize(props, pwds, ips)
        elif type(props) is tuple:
            if pwds and props[0] == 'ENC':
                config[section] = ('ENC', 'REDACTED=',)
            # TODO: Pick numbers for networks
            elif ips and ipv4addr.search(props[0]):
            	config[section] = scrub_ipset(props)

# ---- Difference Engine ----------------------
Undefined = object()

def print_diff_value(name, left, right, prefix="", output=sys.stdout,
        header=None):

    def send_header(name):
        header_is_out = False
        def deferred():
            nonlocal header_is_out
            if not header_is_out:
                if header is not None:
                    header()
                if isinstance(name, Config):
                    output.write("  %sconfig %s\n" % (prefix, name))
                else:
                    output.write("  %sedit %s\n" % (prefix, quote(name,
                        force=not is_number(name))))
                header_is_out = True
        return deferred

    diffs = 0
    if type(left) is dict:
        changes = print_diff_section(left, right, prefix + "    ",
            output, send_header(name))
        if changes > 0:
            if isinstance(name, Config):
                output.write("  %send\n" % (prefix,))
            else:
                output.write("  %snext\n" % (prefix,))
        diffs += changes
    elif left != right:
        if header is not None:
            header()
        diffs += 1
        if left is None:
            output.write("- %sunset %s\n" % (prefix, name,))
        elif left is not Undefined:
            output.write("- %sset %s %s\n" % (prefix, name, " ".join(
quote(x) for x in left)))

        if right is None:
            output.write("+ %sunset %s\n" % (prefix, name,))
        elif right is not Undefined:
            output.write("+ %sset %s %s\n" % (prefix, name, " ".join(
quote(x) for x in right)))

    return diffs

def print_diff_section(left, right, prefix="", output=sys.stdout,
        header=None):

    diffs = 0
    for name, value in left.items():
        if right is Undefined or name not in right:
            rvalue = Undefined
        else:
            rvalue = right[name]

        diffs += print_diff_value(name, value, rvalue, prefix, output, header)

    # And for all the keys missing in the left (or added in the right)
    if type(right) is dict:
        for k in set(right.keys()) - set(left.keys()):
            rvalue = right[k]
            lvalue = None
            if type(rvalue) is dict:
                lvalue = dict((j, Undefined) for j in rvalue)
            diffs += print_diff_value(k, lvalue, rvalue, prefix, output, header)

    return diffs

def print_diffs(configs, output=sys.stdout):
    if len(configs) == 1:
        raise Exception('At least two configs required for diffing')
    left, right = configs[0], configs[1]

    print_diff_section(left, right, "", output)

# ---- Argument Handling ----------------------

section_maps = {
    'ngfw':         ('antivirus profile', 'webfilter urlfilter',
        'webfilter profile'),
    'addresses':    ('firewall address', 'firewall addrgrp'),
    'services':     ('firewall service category', 'firewall service custom',
        'firewall service group',),
    'firewall':     ('firewall shaper traffic-shaper',
        'firewall shaper per-ip-shaper', 'firewall schedule recurring',
        'firewall ippool', 'firewall ldb-monitor', 'firewall vip',
        'firewall profile-protocol-options', 'firewall ssl-ssh-profile',
        'firewall policy',),
    'vpns':         ('vpn ipsec phase1-interface', 'vpn ipsec phase2-interface'),
    'routes':       ('router static',),
}

parser = argparse.ArgumentParser(description='Manipulate multiple Fortigate configs')
parser.add_argument('file', type=str, nargs='+',
    help='A list of files to merge')
parser.add_argument('--vdom', type=str,
    help="Target VDOM for incoming config file. Useful for merging a global "
         "config into a VDOM. Must already exist on the left-most.")
parser.add_argument('--merge', type=str, nargs='+',
    help="List of sections to merge", default=[],
    choices=section_maps.keys())
parser.add_argument('--diff', default=False, action='store_true',
    help="Produce a report of differences between the configs")
parser.add_argument('--get', type=str, help="Extract a single section from the "
    "configuration. Use the text after the token `config`, so `firewall policy` "
    "for instance")
parser.add_argument('--replace',
    default=False, action='store_true',
    help="Configuration of left is replaced with the RHS")
parser.add_argument('--sanitize', type=str, default=[],
    help="Strip passwords and pick random network numbers", nargs="+",
    choices=['pwds','ips'])
parser.add_argument('--ifmap', nargs="+",
    help="Map interface names, eg --ifmap port1:lan1 where `port1` is valid "
        "in the right-hand config and `lan1` should be used in the left-hand "
        "config. Can accept multiple maps separated by space")
parser.add_argument('--csv', action="store_true",
    help="Used with --get, return results as a CSV file")

# ---- The main thing -------------------------

def main():
    args = parser.parse_args()

    configs = []
    # Parse all the configurations
    for file in args.file:
        if file == "-":
            file = sys.stdin
        else:
            file = open(file, "rt")
        sys.stderr.write("# Parsing configuration : %s\n" % (file.name,))
        configs.append(parse_config(file))

    # Translate interface names
    if args.ifmap:
        ifmap = dict(m.split(':', 1) for m in args.ifmap)
        configs[1:] = [translate_interfaces(x, ifmap) for x in configs[1:]]

    # Perform merging as requested
    for s in section_maps.keys():
        if s not in args.merge:
            continue
        sys.stderr.write(">>> Merging %s\n" % (s,))
        l, right = configs[0], configs[1:]

        # Support VDOM option
        if args.vdom is not None and 'vdom' in l:
            l = l.get('vdom').get(args.vdom)
        for name in section_maps[s]:
            sys.stderr.write("    ... section %s\n" % (name,))
            for r in right:
                if args.vdom and 'vdom' in r:
                    r = r.get('vdom').get(args.vdom) or r
                l[Config(name)] = merge_section(l, r, name)

    if args.diff:
        return print_diffs(configs)

    # Fetch the left-most config
    left = configs[0]

    if args.sanitize:
        sanitize(left, 'pwds' in args.sanitize, 'ips' in args.sanitize)

    # Pull the requested section, if any
    if args.get is not None:
        if args.vdom is not None and 'vdom' in left:
            left = left.get('vdom').get(args.vdom)
        left = {Config(args.get): left.get(args.get)}

    # Output the left-most config
    if not args.csv:
        pretty_print_config(left)
    elif args.get:
        print_csv(left.get(args.get))
    else:
        sys.stderr.write("!!! Must use --get with --csv\n")

if __name__ == '__main__':
    main()
