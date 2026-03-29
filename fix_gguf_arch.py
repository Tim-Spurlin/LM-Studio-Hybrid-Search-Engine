import sys
import os
from pathlib import Path
from gguf import GGUFReader, GGUFWriter, Keys
from gguf.scripts.gguf_new_metadata import copy_with_new_metadata, get_field_data, MetadataDetails

def copy_with_renamed_keys(reader, writer):
    new_metadata = {}
    remove_metadata = []
    
    # Pre-process fields to rename llama-embed.* to llama.*
    for field in reader.fields.values():
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == 9 else None  # ARRAY enum
        
        val = MetadataDetails(val_type, field.contents(), sub_type=sub_type)
        
        if field.name.startswith("llama-embed."):
            new_name = field.name.replace("llama-embed.", "llama.")
            new_metadata[new_name] = val
            remove_metadata.append(field.name)
            
    copy_with_new_metadata(reader, writer, new_metadata, remove_metadata)

def main():
    input_file = "/home/saphyre-solutions/Downloads/llama-nemotron-embed-1b-v2.Q8_0.gguf"
    output_file = os.environ.get("EMBED_MODEL_PATH", "./models/llama-nemotron-embed-1b-v2-fixed.gguf")
    
    if os.path.exists(output_file):
        os.remove(output_file)
        
    print(f"Reading {input_file}...")
    reader = GGUFReader(input_file, 'r')
    
    # FORCED OVERRIDE: llama-embed -> llama
    arch = "llama"
    print(f"Setting general.architecture to: {arch}")
    
    writer = GGUFWriter(output_file, arch=arch, endianess=reader.endianess)
    
    alignment = get_field_data(reader, Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment
        
    print(f"Cloning immutable tensors and patching metadata to {output_file}...")
    copy_with_renamed_keys(reader, writer)
    print(f"\n[+] Successfully generated patched GGUF: {output_file}")

if __name__ == "__main__":
    main()
