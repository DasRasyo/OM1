  from dataclasses import dataclass
  from enum import Enum
  from types import ModuleType
  from unittest.mock import patch

  import pytest

  from actions import describe_action, load_action
  from actions.base import ActionConfig, ActionConnector, AgentAction, Interface


  class SampleEnum(str, Enum):
      OPTION_A = "option_a"
      OPTION_B = "option_b"


  @dataclass
  class EnumInput:
      action: SampleEnum


  @dataclass
  class EnumOutput:
      action: SampleEnum


  @dataclass
  class EnumInterface(Interface[EnumInput, EnumOutput]):
      """This action performs an enum-based operation."""

      input: EnumInput
      output: EnumOutput


  @dataclass
  class StringInput:
      action: str


  @dataclass
  class StringOutput:
      action: str


  @dataclass
  class StringInterface(Interface[StringInput, StringOutput]):
      """This action performs a string-based operation."""

      input: StringInput
      output: StringOutput


  class MockConnector(ActionConnector[ActionConfig, StringOutput]):
      def __init__(self, config: ActionConfig):
          super().__init__(config)

      async def connect(self, output_interface: StringOutput) -> None:
          pass


  def _make_interface_module(interface_cls, name="mock_interface"):
      """Helper to create a mock module containing an Interface subclass."""
      mod = ModuleType(name)
      setattr(mod, interface_cls.__name__, interface_cls)
      return mod


  def _make_connector_module(connector_cls, config_cls=None, name="mock_connector"):
      """Helper to create a mock module containing a Connector and optional Config."""
      mod = ModuleType(name)
      setattr(mod, connector_cls.__name__, connector_cls)
      if config_cls is not None:
          setattr(mod, config_cls.__name__, config_cls)
      return mod


  class TestDescribeAction:
      """Tests for the describe_action function."""

      def test_describe_action_excluded(self):
          """Actions with exclude_from_prompt=True should return empty string."""
          config = {"type": "sample", "exclude_from_prompt": True}
          result = describe_action(config)
          assert result == ""

      def test_describe_action_enum_field(self):
          """Enum fields should list allowed values in the description."""
          interface_mod = _make_interface_module(EnumInterface)

          with patch("importlib.import_module", return_value=interface_mod):
              result = describe_action({"type": "sample"})

          assert "OPTION_A" in result
          assert "OPTION_B" in result

      def test_describe_action_string_field(self):
          """String fields should show type annotation in the description."""
          interface_mod = _make_interface_module(StringInterface)

          with patch("importlib.import_module", return_value=interface_mod):
              result = describe_action({"type": "sample"})

          assert "str" in result

      def test_describe_action_includes_docstring(self):
          """Interface docstring should appear in the action description."""
          interface_mod = _make_interface_module(EnumInterface)

          with patch("importlib.import_module", return_value=interface_mod):
              result = describe_action({"type": "sample"})

          assert "enum-based operation" in result

      def test_describe_action_no_interface_found(self):
          """Module without Interface subclass should return empty string."""
          empty_mod = ModuleType("empty")

          with patch("importlib.import_module", return_value=empty_mod):
              result = describe_action({"type": "sample"})

          assert result == ""

      def test_describe_action_invalid_module(self):
          """Non-existent module should return empty string."""
          with patch("importlib.import_module", side_effect=ModuleNotFoundError):
              result = describe_action({"type": "nonexistent"})

          assert result == ""


  class TestLoadAction:
      """Tests for the load_action function."""

      def test_load_action_success(self):
          """Successful load should return an AgentAction with correct attributes."""
          interface_mod = _make_interface_module(StringInterface)
          connector_mod = _make_connector_module(MockConnector)

          with patch("importlib.import_module") as mock_import:
              mock_import.side_effect = lambda name: (
                  interface_mod if "interface" in name else connector_mod
              )
              action = load_action(
                  {"type": "sample", "llm_label": "test_label", "name": "test_name"}
              )

          assert isinstance(action, AgentAction)
          assert action.name == "test_name"
          assert action.llm_label == "test_label"

      def test_load_action_with_custom_config(self):
          """Custom config class from connector module should be used."""

          class CustomConfig(ActionConfig):
              pass

          interface_mod = _make_interface_module(StringInterface)
          connector_mod = _make_connector_module(
              MockConnector, config_cls=CustomConfig
          )

          with patch("importlib.import_module") as mock_import:
              mock_import.side_effect = lambda name: (
                  interface_mod if "interface" in name else connector_mod
              )
              action = load_action(
                  {"type": "sample", "llm_label": "label", "name": "name"}
              )

          assert isinstance(action, AgentAction)

      def test_load_action_no_connector_raises(self):
          """Module without ActionConnector subclass should raise ValueError."""
          interface_mod = _make_interface_module(StringInterface)
          empty_mod = ModuleType("empty")

          with (
              patch("importlib.import_module") as mock_import,
              pytest.raises(ValueError, match="connector"),
          ):
              mock_import.side_effect = lambda name: (
                  interface_mod if "interface" in name else empty_mod
              )
              load_action({"type": "sample", "llm_label": "label", "name": "name"})

      def test_load_action_no_interface_raises(self):
          """Module without Interface subclass should raise an error."""
          empty_mod = ModuleType("empty")
          connector_mod = _make_connector_module(MockConnector)

          with (
              patch("importlib.import_module") as mock_import,
              pytest.raises(Exception),
          ):
              mock_import.side_effect = lambda name: (
                  empty_mod if "interface" in name else connector_mod
              )
              load_action({"type": "sample", "llm_label": "label", "name": "name"})

      def test_load_action_default_exclude_from_prompt(self):
          """Default exclude_from_prompt should be False."""
          interface_mod = _make_interface_module(StringInterface)
          connector_mod = _make_connector_module(MockConnector)

          with patch("importlib.import_module") as mock_import:
              mock_import.side_effect = lambda name: (
                  interface_mod if "interface" in name else connector_mod
              )
              action = load_action(
                  {"type": "sample", "llm_label": "label", "name": "name"}
              )

          assert action.exclude_from_prompt is False

      def test_load_action_exclude_from_prompt_true(self):
          """exclude_from_prompt=True should be passed to AgentAction."""
          interface_mod = _make_interface_module(StringInterface)
          connector_mod = _make_connector_module(MockConnector)

          with patch("importlib.import_module") as mock_import:
              mock_import.side_effect = lambda name: (
                  interface_mod if "interface" in name else connector_mod
              )
              action = load_action(
                  {
                      "type": "sample",
                      "llm_label": "label",
                      "name": "name",
                      "exclude_from_prompt": True,
                  }
              )

          assert action.exclude_from_prompt is True
