
import json
from tools.wiki import search as wiki_search
from tools.serpapi import search as google_search
from typing import Callable, Union, Dict, List
from enum import Enum, auto
from pydantic import BaseModel, Field
from myutils.file_ops import read_text_file, write_to_file
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

Observation = Union[str, Exception]

PROMPT_TEMPLATE_PATH = "./prompt_templates/ReAct.txt"
OUTPUT_TRACE_PATH = "./output_traces/traces.txt"

class Name(Enum):
    """Enumerate on tool names available to agent"""

    WIKIPEDIA = auto()
    GOOGLE = auto()
    NONE = auto()

    def __str__(self) -> str:
        """
        String representation of tool names
        :return: string
        """
        return self.name.lower()

class Choice(BaseModel):
    """
    Represents a choice of tool with a reason for selection
    """
    name: Name = Field(..., description="The name of the tool chosen.")
    reason: str = Field(..., description="The reason for choosing this tool.")

class Message(BaseModel):
    """
    Represents a message with sender role and context
    """
    role: str = Field(..., description="the role of the message sender")
    content: str = Field(..., description="The content of the message")

class Tool:
    """
    A wrapper class for tools used by the agent, executing a function based on tool type
    """
    def __init__(self, name: Name, function: Callable[[str], str]):
        """
        Initializes a Tool with a name and an associated function
        :param name: The Name of the tool
        :param function: The function associated with the tool
        """
        self.name = name
        self.func = function

    def use(self, query: str) -> Observation:
        """
        Executes the tool's function with the provided query
        :param query: The input query for the tool
        :return: Observation: Result from tool's function or an error message if and exception occurs
        """
        try:
            return self.func(query)
        except Exception as e:
            print(f"Error executing tool {self.name}: {e}")
            return str(e)

class Agent:
    """
    Defines the agent responsible for executing queries and handling tool interactions.
    """
    def __init__(self, model: str) -> None:
        """
        Initializes the Agent with a generative model, tools dictionary, and a message log
        :param model: The Generative model to be used by agent
        """
        self.model = model
        self.tools: Dict[Name, Tool] = dict()
        self.messages: List[Message] = list()
        self.query = ""
        self.max_iterations = 5
        self.current_iteration = 0
        self.template = self.load_template()

    def load_template(self) -> str:
        """
        Loads the prompt template from a file
        :return: the content of the prompt template file
        """
        return read_text_file(PROMPT_TEMPLATE_PATH)

    def register(self, name: Name, func: Callable[[str], str]) -> Name:
        """
        Registers a tool to the agent
        :param name:
        :param func:
        :return:
        """
        self.tools[name] = Tool(name, func)

    def trace(self, role: str, content: str) -> None:
        """
        Logs the message with the specified role and content and writes to a file
        :param role:
        :param content:
        :return:
        """
        if role != "system":
            self.messages.append(Message(role=role, content=content))
        write_to_file(OUTPUT_TRACE_PATH, content=f"{role}: {content}\n")

    def get_history(self) -> str:
        """
        Retrieves the conversation history
        :return: Formatted history of messages
        """
        return "\n".join([f"{message.role}: {message.content}" for message in self.messages])

    def think(self) -> None:
        """
        Processes the current query, decides actions and iterates until a solution or max iteration limit is reached
        """
        self.current_iteration += 1
        write_to_file(OUTPUT_TRACE_PATH, content=f"\n{'='*50}\nIteration {self.current_iteration}\n{'='*50}\n")

        if self.current_iteration > self.max_iterations:
            self.trace("assistant", "I'm sorry, but I couldn't find a satisfactory answer within the allowed number of iterations. Here's what I know so far: " + self.get_history())
            return
        prompt = self.template.format(
            query = self.query,
            history = self.get_history(),
            tools = ", ".join([str(tool.name) for tool in self.tools.values()])
        )

        response = self.ask_gemini(prompt)
        print(f"Thinking => {response}")
        self.trace("assistant", f"Thought: {response}")
        self.decide(response)

    def decide(self, response: str) -> None:
        """
        Processes the agent's response, deciding actions or final answers
        """
        try:
            cleaned_response = response.strip().strip('`').strip()
            if cleaned_response.startswith('json'):
                cleaned_response = cleaned_response[4:].strip()
            parsed_response = json.loads(cleaned_response)

            if "action" in parsed_response:
                action = parsed_response['action']
                tool_name = Name[action["name"].upper()]
                if tool_name == Name.NONE:
                    print("No action needed. Proceeding to final answer.")
                    self.think()
                else:
                    self.trace("assistant", f"Action: Using {tool_name} tool")
                    self.act(tool_name, action.get("input", self.query))
            elif "answer" in parsed_response:
                self.trace("assistant", f"Final Answer: {parsed_response['answer']}")
            else:
                raise ValueError("Invalid response format")

        except json.JSONDecodeError as e:
            print(f"Failed to parse response: {response}. Error: {str(e)}")
            self.trace("assistant", "I encountered an error in processing. Let me try again.")
            self.think()
        except Exception as e:
            print(f"Error processing response: {str(e)}")
            self.trace("assistant", "I encountered an unexpected error. Let me try a different approach.")
            self.think()

    def act(self, tool_name: Name, query: str) -> None:
        """
        Executes specified tool function on the query
        :param tool_name:
        :param query:
        :return:
        """
        tool = self.tools.get(tool_name)
        if tool:
            result = tool.use(query)
            observation = f"Observation from {tool_name}: {result}"
            self.trace("system", observation)
            self.messages.append(Message(role="system", content=observation))
            self.think()
        else:
            print(f"No tool registered for choice: {tool_name}")
            self.trace("system", f"Error: Tool {tool_name} not found")
            self.think()

    def execute(self, query) -> str:
        """
        Executes the agent's query-processing workflow
        :param query:
        :return:
        """
        self.query = query
        self.trace(role="user", content=query)
        self.think()
        return self.messages[-1].content

    def ask_gemini(self, prompt: str) -> str:
        """
        Queries the generative model with a prompt
        :param prompt:
        :return:
        """
        llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0.0,
            max_tokens=None,
            timeout=None,
            max_retries=2

        )

        response = llm.invoke(prompt).content
        return str(response) if response is not None else "No response from Gemini"


def run(query: str) -> str:
    """
    Sets up the agent, registers tools and executes a query
    :param query:
    :return:
    """
    # gemini = GenerativeModel(os.getenv("MODEL_NAME"))
    agent = Agent(model="gemini-1.5-flash")
    agent.register(Name.WIKIPEDIA, wiki_search)
    agent.register(Name.GOOGLE, google_search)

    answer = agent.execute(query)
    return answer


if __name__ == "__main__":
    query = "what are the top 3 contries that faced the highest levels of nuclear radiation in the history of mankind and what are their current president's names?"
    final_answer = run(query)
    print(final_answer)