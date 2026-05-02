"""Tests pour setup_indexes.py — creation d'index MongoDB."""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from collection.setup_indexes import setup_indexes, setup_schema_validation


class TestSetupIndexes:
    def test_creates_all_indexes(self):
        mock_db = MagicMock()
        mock_col = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_col)
        mock_col.create_index.return_value = "test_index"

        setup_indexes(mock_db)

        assert mock_col.create_index.call_count == 7
        # Verify unique index on uri
        calls = mock_col.create_index.call_args_list
        uri_call = calls[0]
        assert uri_call.kwargs.get('unique') is True
        assert uri_call.kwargs.get('name') == 'idx_uri_unique'

    def test_creates_ttl_index(self):
        mock_db = MagicMock()
        mock_col = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_col)
        mock_col.create_index.return_value = "test_index"

        setup_indexes(mock_db)

        calls = mock_col.create_index.call_args_list
        # TTL index is the 7th call (index 6)
        ttl_call = calls[6]
        assert ttl_call.kwargs.get('expireAfterSeconds') == 31536000
        assert ttl_call.kwargs.get('name') == 'idx_collected_at_ttl_12months'


class TestSchemaValidation:
    def test_applies_schema(self):
        mock_db = MagicMock()
        setup_schema_validation(mock_db)
        mock_db.command.assert_called_once()
        args = mock_db.command.call_args
        assert args[0][0] == "collMod"
        assert args[0][1] == "raw_posts"
        assert 'validator' in args.kwargs
        assert args.kwargs['validationLevel'] == 'moderate'
