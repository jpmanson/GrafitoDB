import os
import sys
import zstandard as zstd
import tarfile

def extract_dump(dump_path, output_dir):
    if not os.path.exists(dump_path):
        print(f"Dump file not found: {dump_path}")
        return

    print(f"Extracting {dump_path} to {output_dir}...")
    
    with open(dump_path, "rb") as handle:
        header = handle.read(4)
        if header == b"DZV1":
            offset = 4
        elif header == b"DGV1":
            print("Unsupported DGV1 (gzip) dump format.")
            return
        else:
            offset = 0

    os.makedirs(output_dir, exist_ok=True)
    with open(dump_path, "rb") as handle:
        handle.seek(offset)
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(handle)
        try:
            # Skip 24 bytes? The original code does reader.read(24)
            # Let's see if that's correct.
            reader.read(24) 
            
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    print(f"Found member: {member.name}")
                    name = member.name.lstrip("./")
                    # Extract everything to see what we have
                    target_path = os.path.join(output_dir, name)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    if member.isdir():
                        os.makedirs(target_path, exist_ok=True)
                        continue
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    with open(target_path, "wb") as out_handle:
                        out_handle.write(extracted.read())
        finally:
            reader.close()

if __name__ == "__main__":
    dump_path = "examples/datasets/recommendations-5.26.dump"
    output_dir = "debug_neo4j"
    extract_dump(dump_path, output_dir)
    
    print("\nExtracted files:")
    for root, dirs, files in os.walk(output_dir):
        for name in files:
            path = os.path.join(root, name)
            size = os.path.getsize(path)
            print(f"{path} - {size} bytes")
