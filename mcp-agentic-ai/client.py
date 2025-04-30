import json
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage, AssistantMessage
from gen_ai_hub.orchestration.models.template import Template
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.service import OrchestrationService
from gen_ai_hub.orchestration.models.response_format import ResponseFormatJsonSchema


class MCPAgentExecutor:
    def __init__(self, llm, mcp_session: ClientSession, verbose=True):
        self.llm = llm
        self.session = mcp_session
        self.verbose = verbose

    def _build_dynamic_schema(self):
        return {
            "title": "ToolCalls",
            "type": "object",
            "properties": {
                "tool_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "decision": {"type": "string"},
                            "reason": {"type": "string"},
                            "function": {"type": "string"},
                            "parameters": {"type": "object"}
                        },
                        "required": ["decision", "reason", "function", "parameters"]
                    }
                }
            },
            "required": ["tool_calls"]
        }

    async def _generate_instruction(self):
        description = json.dumps(await self.list_tools(), indent=2)
        return f"""
        You are an intelligent AI assistant capable of deciding whether to invoke tools based on the user's request.

        Available tools:
        {description}

        Instructions:
        - For each relevant tool, return a JSON entry with the function name and parameters.
        - If no tool is relevant, return an entry with decision = "no_tool".

        Return ONLY valid JSON like:
        {{
        "tool_calls": [
            {{
            "decision": "tool",
            "reason": "The user asked for weather.",
            "function": "get_weather",
            "parameters": {{
                "latitude": 48.8566,
                "longitude": 2.3522
            }}
            }},
            {{
            "decision": "tool",
            "reason": "The user asked for time.",
            "function": "get_time_now",
            "parameters": {{}}
            }},
            {{
            "decision": "tool",
            "reason": "The user asked a question that requires retrieving information.",
            "function": "retriever",
            "parameters": {{
                "question": "What is SAP Datasphere?"
            }}
            }}
        ]
        }}
        """

    async def list_tools(self):
        tools_result = await self.session.list_tools()
        return {tool.name: {"description": tool.description} for tool in tools_result.tools}    


    async def _execute_tool(self, decision):
        func = decision["function"]
        args = decision.get("parameters", {})

        try:
            result = await self.session.call_tool(func, arguments=args)
            if self.verbose:
                print(f"\nTool '{func}' executed. Result: {result.content[0].text}")
            return result.content[0].text
        except Exception as e:
            return f"Error: {str(e)}"
    
    
    async def _finalize_response(self, original_query, tool_results, messages):
        # Append summary and results to LLM context
        # Give explicit instruction and reinforce tool results context
        messages.append(SystemMessage(
            """ 
            You now have access to the results provided by the tools. When the results are clear and complete, use only that information to answer the user's 
            question in a natural, helpful, and concise manner. However, if any result appears vague, incomplete, or states uncertainty (e.g., "I don't know"), 
            rely on your own knowledge to deliver an accurate and informative response.
            Always avoid requesting information already provided. Focus on clarity, relevance, and user value.

            """            
        ))

        messages.append(UserMessage(f"User question: {original_query}"))

        # Structured, clean summary of tool outputs
        tool_summary = "\n".join(
            [f"- Tool `{name}` returned: {json.dumps(result)}" for name, result in tool_results]
        )
        messages.append(UserMessage(f"Tool Results:\n{tool_summary}"))


        # Final orchestration
        template = Template(messages=messages, response_format="text")
        config = OrchestrationConfig(template=template, llm=self.llm)
        response = OrchestrationService(config=config).run()
        return response.module_results.llm.choices[0].message.content    
    

    async def run(self, user_query: str):
        system_message = SystemMessage(await self._generate_instruction())
        prompt = UserMessage(user_query)
        messages = [system_message, prompt]

        template = Template(
            messages=messages,
            response_format=ResponseFormatJsonSchema(
                name="ToolCall",
                description="Tool execution format",
                schema=self._build_dynamic_schema()
            )
        )
        config = OrchestrationConfig(template=template, llm=self.llm)
        response = OrchestrationService(config=config).run()

        decisions_json = json.loads(response.module_results.llm.choices[0].message.content)

        if self.verbose:
            print("\nLLM Reasoning:")
            print(json.dumps(decisions_json, indent=2))

        tool_results = []
        messages = [system_message, prompt]

        for decision in decisions_json.get("tool_calls", []):
            if decision.get("decision") == "tool":
                tool_response = await self._execute_tool(decision)
                tool_results.append((decision["function"], tool_response))
                messages.append(AssistantMessage(json.dumps(decision)))
            else:
                messages.append(AssistantMessage(json.dumps(decision)))

        return await self._finalize_response(user_query, tool_results, messages)


async def main():
    async with sse_client("https://<your-mcp-server>/sse") as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            llm = LLM(name="gpt-4o", version="latest", parameters={"max_tokens": 2000, "temperature": 0.2})
            agent = MCPAgentExecutor(llm=llm, mcp_session=session, verbose=True)
            
            prompt = "Can you give me a bread recipe and also tell me: how’s the weather in Brazil, what time it is now, and what is the role of SAP Datasphere in SAP Business Data Cloud?"
            print(f"Prompt: {prompt}")
            result = await agent.run(prompt)
            print("\nFinal Answer:", result)


if __name__ == "__main__":
    asyncio.run(main())
