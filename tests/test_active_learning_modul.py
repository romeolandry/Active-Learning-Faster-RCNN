#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `active_learning_modul` package."""


import unittest
from click.testing import CliRunner

from active_learning_modul import active_learning_modul
from active_learning_modul import cli


class TestActive_learning_modul(unittest.TestCase):
    """Tests for `active_learning_modul` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'active_learning_modul.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
