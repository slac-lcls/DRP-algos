#!/usr/bin/env python3
"""Embed a header's text as a C++ string: embed_header.py <input.hh> <output.cc> <variable>."""
import sys

src: str = sys.argv[1]
out: str = sys.argv[2]
var: str = sys.argv[3]

text: str = open(src).read()
# Delimiter used for C++ raw multi-line string
delim: str = "drpembed"
if f"){delim}\"" in text:
    sys.exit(f"{src} contains the raw-string delimiter ){delim}\"")
with open(out, "w") as f:
    f.write(
        "namespace DrpAlgos {\n"
        f"  extern const char* const {var};\n"
        f"  const char* const {var} = R\"{delim}({text}){delim}\";\n"
        "} // namespace DrpAlgos\n"
    )
