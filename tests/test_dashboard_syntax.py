"""
Syntax validation tests for key application files.

Uses py_compile to verify that the dashboard and main app files
are syntactically valid Python without needing to import all
their dependencies.
"""

import os
import py_compile
import pytest


_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')


class TestSyntaxValidation:
    """Verify that key Python files compile without syntax errors."""

    def test_dashboard_app_compiles(self):
        """dashboard/app.py should be syntactically valid Python."""
        path = os.path.join(_PROJECT_ROOT, 'dashboard', 'app.py')
        if not os.path.exists(path):
            pytest.skip("dashboard/app.py not found")
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as exc:
            pytest.fail(f"dashboard/app.py has syntax errors: {exc}")

    def test_main_app_compiles(self):
        """src/app/main.py should be syntactically valid Python."""
        path = os.path.join(_PROJECT_ROOT, 'src', 'app', 'main.py')
        if not os.path.exists(path):
            pytest.skip("src/app/main.py not found")
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as exc:
            pytest.fail(f"src/app/main.py has syntax errors: {exc}")
