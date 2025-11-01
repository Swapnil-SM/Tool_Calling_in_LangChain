from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import requests
#import os

load_dotenv()


# tool creation

@tool
def multiply(a: int, b: int) -> int:
  """Given 2 numbers a and b this tool returns their product"""
  return a * b

# print(multiply.invoke({'a':3, 'b':4}))
# print(multiply.args)
# print(multiply.description)

llm = ChatGoogleGenerativeAI(model="gemini-pro")

#tool binding
llm_with_tools = llm.bind_tools([multiply])
result =llm_with_tools.invoke('can you multiply 3 with 1000') #content is empty when the model triggers a tool call, not when it gives a normal text response.
result.tool_calls #shows the list of tool calls .
print(result.tool_calls[0]['args']) #shows the arguments with which the tool was called.

#multiply.invoke(result.tool_calls[0])  


query = HumanMessage('can you multiply 3 with 1000')
messages = [query]
result = llm_with_tools.invoke(messages)
messages.append(result)
tool_result = multiply.invoke(result.tool_calls[0])
messages.append(tool_result)  #meesges now has 3 messages, user Humanmessage(query),AImessage(model response with tool call),Toolmessage() toolmessage(result)
final_response=llm_with_tools.invoke(messages) #final response after tool execution
print(final_response.content)
