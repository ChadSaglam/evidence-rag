"""Export checked-in schemas from the repository root."""

from evidence_rag.schema_export import export_schemas

if __name__ == "__main__":
    export_schemas("schemas")
