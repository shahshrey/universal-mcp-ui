import streamlit as st
import asyncio
import json
import os
from typing import List, Tuple
import sys
from contextlib import asynccontextmanager

from mcpcli.config import load_config
from mcpcli.transport.stdio.stdio_client import stdio_client
from mcpcli.messages.send_ping import send_ping
from mcpcli.messages.send_tools_list import send_tools_list
from mcpcli.messages.send_resources import send_resources_list
from mcpcli.messages.send_initialize_message import send_initialize
from mcpcli.chat_handler import handle_chat_mode

# Constants
DEFAULT_CONFIG_FILE = "server_config.json"
PROVIDERS = ["openai", "anthropic", "ollama"]
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-haiku-latest",
    "ollama": "qwen2.5-coder"
}

@asynccontextmanager
async def managed_connection(config_path: str, server_name: str):
    server_params = await load_config(config_path, server_name)
    async with stdio_client(server_params) as (read_stream, write_stream):
        init_result = await send_initialize(read_stream, write_stream)
        if init_result:
            yield read_stream, write_stream
        else:
            st.error(f"Failed to initialize {server_name}")
            yield None, None

async def ensure_connection_async(config_file: str, server_name: str):
    try:
        async with managed_connection(config_file, server_name) as (read_stream, write_stream):
            if read_stream and write_stream:
                # Test connection with ping
                result = await send_ping(read_stream, write_stream)
                if result:
                    return read_stream, write_stream
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
    return None, None

async def handle_chat_async(read_stream, write_stream, prompt: str, provider: str, model: str):
    try:
        server_streams = [(read_stream, write_stream)]
        response = await handle_chat_mode(server_streams, provider, model, prompt)
        return response
    except Exception as e:
        st.error(f"Chat error: {str(e)}")
        return None

async def list_tools_async(read_stream, write_stream):
    try:
        response = await send_tools_list(read_stream, write_stream)
        return response.get("tools", [])
    except Exception as e:
        st.error(f"Error listing tools: {str(e)}")
        return []

async def list_resources_async(read_stream, write_stream):
    try:
        response = await send_resources_list(read_stream, write_stream)
        return response.get("resources", [])
    except Exception as e:
        st.error(f"Error listing resources: {str(e)}")
        return []

def get_server_names(config_file: str) -> List[str]:
    try:
        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)
            return list(config.get("mcpServers", {}).keys())
    except Exception as e:
        st.error(f"Error loading server config: {str(e)}")
        return []

def main():
    st.set_page_config(page_title="MCP Interface", layout="wide")
    st.title("Model Context Provider Interface")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        config_file = st.text_input("Config File Path", value=DEFAULT_CONFIG_FILE)
        server_names = get_server_names(config_file)
        default_server = server_names[0] if server_names else "sqlite"
        server_name = st.selectbox("Server Name", options=server_names, index=0 if server_names else None)
        provider = st.selectbox("Provider", PROVIDERS)
        model = st.text_input("Model", value=DEFAULT_MODELS[provider])

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Tools", "Resources", "System"])

    with tab1:
        st.header("Chat Interface")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What would you like to know?"):
            async def process_chat():
                async with managed_connection(config_file, server_name) as (read_stream, write_stream):
                    if read_stream and write_stream:
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        with st.chat_message("assistant"):
                            response = await handle_chat_async(read_stream, write_stream, prompt, provider, model)
                            if response:
                                st.markdown(response)
                                st.session_state.messages.append({"role": "assistant", "content": response})

            asyncio.run(process_chat())

    with tab2:
        st.header("Tools")
        if st.button("List Tools"):
            async def process_tools():
                async with managed_connection(config_file, server_name) as (read_stream, write_stream):
                    if read_stream and write_stream:
                        tools = await list_tools_async(read_stream, write_stream)
                        for tool in tools:
                            st.write(f"**{tool['name']}**: {tool['description']}")

            asyncio.run(process_tools())

    with tab3:
        st.header("Resources")
        if st.button("List Resources"):
            async def process_resources():
                async with managed_connection(config_file, server_name) as (read_stream, write_stream):
                    if read_stream and write_stream:
                        resources = await list_resources_async(read_stream, write_stream)
                        for resource in resources:
                            st.json(resource)

            asyncio.run(process_resources())

    with tab4:
        st.header("System Status")
        if st.button("Ping Server"):
            async def process_ping():
                async with managed_connection(config_file, server_name) as (read_stream, write_stream):
                    if read_stream and write_stream:
                        result = await send_ping(read_stream, write_stream)
                        if result:
                            st.success("Server is responsive")
                        else:
                            st.error("Server is not responding")

            asyncio.run(process_ping())

if __name__ == "__main__":
    main() 