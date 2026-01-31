"""
Spider Dataset Loader

Loads and parses the Spider dataset files (train, dev, test) along with
table schemas. Provides clean access to examples and database metadata.

Author: Academic NL2SQL Project
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SpiderExample:
    """
    Represents a single Spider dataset example.
    
    Attributes:
        db_id: Database identifier
        question: Natural language question
        query: Target SQL query (ground truth)
        query_toks: Tokenized SQL query
        query_toks_no_value: Tokenized SQL with values replaced
        question_toks: Tokenized natural language question
        sql: Parsed SQL structure (dict)
    """
    db_id: str
    question: str
    query: str
    query_toks: List[str]
    query_toks_no_value: List[str]
    question_toks: List[str]
    sql: Dict


@dataclass
class TableSchema:
    """
    Represents database schema information.
    
    Attributes:
        db_id: Database identifier
        table_names: List of table names in the database
        table_names_original: Original (case-sensitive) table names
        column_names: List of [table_idx, column_name] pairs
        column_names_original: Original (case-sensitive) column names
        column_types: List of column data types
        primary_keys: List of primary key column indices
        foreign_keys: List of [from_col_idx, to_col_idx] pairs
    """
    db_id: str
    table_names: List[str]
    table_names_original: List[str]
    column_names: List[Tuple[int, str]]
    column_names_original: List[Tuple[int, str]]
    column_types: List[str]
    primary_keys: List[int]
    foreign_keys: List[Tuple[int, int]]


class SpiderDataLoader:
    """
    Loads Spider dataset files and provides access to examples and schemas.
    """
    
    def __init__(self, spider_data_dir: str):
        """
        Initialize the Spider data loader.
        
        Args:
            spider_data_dir: Path to the spider_data directory
        """
        self.spider_data_dir = Path(spider_data_dir)
        self._validate_data_dir()
        
        # Cache for loaded data
        self._schemas: Optional[Dict[str, TableSchema]] = None
        self._train_data: Optional[List[SpiderExample]] = None
        self._dev_data: Optional[List[SpiderExample]] = None
        self._test_data: Optional[List[SpiderExample]] = None
    
    def _validate_data_dir(self) -> None:
        """
        Validate that the Spider data directory exists and contains required files.
        
        Raises:
            FileNotFoundError: If directory or required files are missing
        """
        if not self.spider_data_dir.exists():
            raise FileNotFoundError(
                f"Spider data directory not found: {self.spider_data_dir}"
            )
        
        required_files = [
            "train_spider.json",
            "dev.json",
            "tables.json"
        ]
        
        for filename in required_files:
            filepath = self.spider_data_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Required file not found: {filepath}"
                )
    
    def load_schemas(self) -> Dict[str, TableSchema]:
        """
        Load database schemas from tables.json.
        
        Returns:
            Dictionary mapping db_id to TableSchema objects
        """
        if self._schemas is not None:
            return self._schemas
        
        tables_path = self.spider_data_dir / "tables.json"
        
        with open(tables_path, 'r', encoding='utf-8') as f:
            tables_data = json.load(f)
        
        self._schemas = {}
        
        for table_entry in tables_data:
            schema = TableSchema(
                db_id=table_entry["db_id"],
                table_names=table_entry["table_names"],
                table_names_original=table_entry["table_names_original"],
                column_names=[(item[0], item[1]) for item in table_entry["column_names"]],
                column_names_original=[(item[0], item[1]) for item in table_entry["column_names_original"]],
                column_types=table_entry["column_types"],
                primary_keys=table_entry["primary_keys"],
                foreign_keys=[(fk[0], fk[1]) for fk in table_entry["foreign_keys"]]
            )
            self._schemas[schema.db_id] = schema
        
        return self._schemas
    
    def _load_examples(self, filepath: Path) -> List[SpiderExample]:
        """
        Load examples from a Spider JSON file.
        
        Args:
            filepath: Path to the JSON file (train/dev/test)
        
        Returns:
            List of SpiderExample objects
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = []
        
        for item in data:
            example = SpiderExample(
                db_id=item["db_id"],
                question=item["question"],
                query=item["query"],
                query_toks=item["query_toks"],
                query_toks_no_value=item["query_toks_no_value"],
                question_toks=item["question_toks"],
                sql=item["sql"]
            )
            examples.append(example)
        
        return examples
    
    def load_train(self) -> List[SpiderExample]:
        """
        Load training examples.
        
        Returns:
            List of training SpiderExample objects
        """
        if self._train_data is not None:
            return self._train_data
        
        train_path = self.spider_data_dir / "train_spider.json"
        self._train_data = self._load_examples(train_path)
        return self._train_data
    
    def load_dev(self) -> List[SpiderExample]:
        """
        Load development (validation) examples.
        
        Returns:
            List of development SpiderExample objects
        """
        if self._dev_data is not None:
            return self._dev_data
        
        dev_path = self.spider_data_dir / "dev.json"
        self._dev_data = self._load_examples(dev_path)
        return self._dev_data
    
    def load_test(self) -> List[SpiderExample]:
        """
        Load test examples (if available).
        
        Returns:
            List of test SpiderExample objects
        
        Raises:
            FileNotFoundError: If test.json does not exist
        """
        if self._test_data is not None:
            return self._test_data
        
        test_path = self.spider_data_dir / "test.json"
        
        if not test_path.exists():
            raise FileNotFoundError(
                f"Test file not found: {test_path}"
            )
        
        self._test_data = self._load_examples(test_path)
        return self._test_data
    
    def get_schema(self, db_id: str) -> Optional[TableSchema]:
        """
        Get schema for a specific database.
        
        Args:
            db_id: Database identifier
        
        Returns:
            TableSchema object or None if not found
        """
        if self._schemas is None:
            self.load_schemas()
        
        return self._schemas.get(db_id)
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with counts for train, dev, and total schemas
        """
        train_data = self.load_train()
        dev_data = self.load_dev()
        schemas = self.load_schemas()
        
        stats = {
            "train_examples": len(train_data),
            "dev_examples": len(dev_data),
            "total_databases": len(schemas),
            "unique_train_databases": len(set(ex.db_id for ex in train_data)),
            "unique_dev_databases": len(set(ex.db_id for ex in dev_data))
        }
        
        return stats


def load_spider_dataset(spider_data_dir: str) -> Tuple[List[SpiderExample], List[SpiderExample], Dict[str, TableSchema]]:
    """
    Convenience function to load all Spider data at once.
    
    Args:
        spider_data_dir: Path to the spider_data directory
    
    Returns:
        Tuple of (train_examples, dev_examples, schemas)
    """
    loader = SpiderDataLoader(spider_data_dir)
    
    train_data = loader.load_train()
    dev_data = loader.load_dev()
    schemas = loader.load_schemas()
    
    return train_data, dev_data, schemas


if __name__ == "__main__":
    # Example usage for testing
    import sys
    
    if len(sys.argv) > 1:
        spider_dir = sys.argv[1]
    else:
        # Default path (relative to project root)
        spider_dir = "../../spider_data"
    
    print(f"Loading Spider dataset from: {spider_dir}")
    
    try:
        loader = SpiderDataLoader(spider_dir)
        
        # Load and print statistics
        stats = loader.get_statistics()
        print("\n📊 Dataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Show a sample example
        train_data = loader.load_train()
        if train_data:
            print("\n📝 Sample Training Example:")
            sample = train_data[0]
            print(f"  Database: {sample.db_id}")
            print(f"  Question: {sample.question}")
            print(f"  SQL Query: {sample.query}")
        
        # Show a sample schema
        schemas = loader.load_schemas()
        if train_data:
            sample_db_id = train_data[0].db_id
            schema = loader.get_schema(sample_db_id)
            if schema:
                print(f"\n🗂️  Sample Schema ({sample_db_id}):")
                print(f"  Tables: {', '.join(schema.table_names)}")
                print(f"  Total Columns: {len(schema.column_names)}")
        
        print("\n✅ Data loading successful!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
