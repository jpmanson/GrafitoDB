import os
import tempfile
import json
from grafito.importers.neo4j_dump import extract_dump, find_store_dir, Neo4jStoreParser

def debug_dump_parsing(dump_path: str, limit: int = 100):
    """
    Extracts a Neo4j dump and prints verbose details for a limited number
    of nodes and relationships to debug the parsing logic.
    """
    if not os.path.exists(dump_path):
        print(f"Dump file not found: {dump_path}")
        return

    # Use a persistent temp dir to avoid re-extracting
    temp_dir = os.path.join(os.getcwd(), "tmp_debug_parser")
    print(f"Using temporary directory: {temp_dir}")

    if not os.path.exists(os.path.join(temp_dir, "neostore.nodestore.db")):
        print(f"Extracting dump file to {temp_dir}...")
        extract_dump(dump_path, temp_dir)
    else:
        print("Dump files already extracted.")

    store_dir = find_store_dir(temp_dir)
    print(f"Found store directory: {store_dir}")

    parser = None
    try:
        parser = Neo4jStoreParser(store_dir, endian=">")
        
        print("\n--- DEBUGGING NODES (LIMIT 100) ---")
        for i, node_data in enumerate(parser.parse_nodes()):
            if i >= limit:
                break
            
            print(f"\n[Node ID: {node_data['id']}]")
            print(f"  Labels: {node_data.get('labels', [])}")
            print("  Properties:")
            # Pretty print properties
            props = node_data.get("properties", {})
            if props:
                for key, value in props.items():
                    print(f"    - {key}: {value} ({type(value).__name__})")
            else:
                print("    (None)")

        print("\n\n--- DEBUGGING RELATIONSHIPS (LIMIT 100) ---")
        for i, rel_data in enumerate(parser.parse_relationships()):
            if i >= limit:
                break
            
            print(f"\n[Relationship ID: {rel_data['id']}]")
            print(f"  Start Node: {rel_data['start_node']}")
            print(f"  End Node: {rel_data['end_node']}")
            print(f"  Type: {rel_data.get('rel_type')}")
            print("  Properties:")
            props = rel_data.get("properties", {})
            if props:
                for key, value in props.items():
                    print(f"    - {key}: {value} ({type(value).__name__})")
            else:
                print("    (None)")

    except Exception as e:
        print(f"\nAn error occurred during parsing: {e}")
    finally:
        if parser:
            parser.close()
        print("\n--- DEBUGGING COMPLETE ---")


if __name__ == "__main__":
    DUMP_FILE = "examples/datasets/recommendations-5.26.dump"
    debug_dump_parsing(DUMP_FILE)
