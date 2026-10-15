#!/usr/bin/env python3
"""Compile the supplied manuscript ZIP without editing the original source."""
import argparse
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--tectonic", default="tectonic")
    parser.add_argument("--output", type=Path, default=root / "docs/paper.pdf")
    args = parser.parse_args()
    if shutil.which(args.tectonic) is None:
        parser.error("Install Tectonic or pass --tectonic /path/to/tectonic")
    with tempfile.TemporaryDirectory(prefix="sgip-paper-") as tmp:
        work = Path(tmp).resolve()
        with zipfile.ZipFile(args.archive) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                dest = (work / member.filename).resolve()
                if work not in dest.parents:
                    raise ValueError(f"Unsafe archive path: {member.filename}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(archive.read(member))
        source = work / "neurips_2026.tex"
        if not source.is_file():
            parser.error("The archive must contain neurips_2026.tex")
        def omit_placeholder(match):
            active = "\n".join(re.split(r"(?<!\\)%", line)[0]
                               for line in match.group(1).splitlines()).strip()
            if active == "Acknowledgments Part!!":
                return "% Placeholder acknowledgments omitted for the web PDF."
            return match.group(0)
        source.write_text(re.sub(
            r"\\begin\{ack\}(.*?)\\end\{ack\}", omit_placeholder,
            source.read_text(encoding="utf-8"), flags=re.S), encoding="utf-8")
        for file in work.rglob("*.tex"):
            text = file.read_text(encoding="utf-8")
            file.write_text(text.replace("–", "--").replace("—", "---"), encoding="utf-8")
        subprocess.run([args.tectonic, "--outdir", str(work), str(source)], check=True, cwd=work)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(work / "neurips_2026.pdf", args.output)
    print(f"Paper written to {args.output}")


if __name__ == "__main__":
    main()
