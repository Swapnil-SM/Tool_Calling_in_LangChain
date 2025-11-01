# tool create
from langchain_openai import ChatOpenAI
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import requests 
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
  """
  This function fetches the currency conversion factor between a given base currency and a target currency
  """
  url = f'https://v6.exchangerate-api.com/v6/ed6c0a643b83e7c4b1e52dd6/pair/{base_currency}/{target_currency}' 
  #link is taken from exchangerate api.com documentation

  response = requests.get(url) #send HTTP requests (like visiting a web link) and get data from APIs.

  return response.json()


@tool
def convert(base_currency_value: int, conversion_rate:Annotated[float,InjectedToolArg]) -> float:
  """
  given a currency conversion rate this function calculates the target currency value from a given base currency value
  """

  return base_currency_value * conversion_rate

print(get_conversion_factor.invoke({'base_currency':'USD','target_currency':'INR'}))
print(convert.invoke({'base_currency_value':100,'conversion_rate':82.5}))


# tool binding
llm = ChatOpenAI()

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

#Tool Calling
messages = [HumanMessage('What is the conversion factor between INR and USD, and based on that can you convert 10 inr to usd')]
ai_message = llm_with_tools.invoke(messages)
print(ai_message.tool_calls) #details of tool calls made by the model

messages.append(ai_message)


import json

#tool execution
for tool_call in ai_message.tool_calls:
  # execute the 1st tool and get the value of conversion rate
  if tool_call['name'] == 'get_conversion_factor':
    tool_message1 = get_conversion_factor.invoke(tool_call)
    # fetch this conversion rate
    conversion_rate = json.loads(tool_message1.content)['conversion_rate'] #we convert it to dict to fetch the conversion rate value
    # append this tool message to messages list
    messages.append(tool_message1)
  # execute the 2nd tool using the conversion rate from tool 1
  if tool_call['name'] == 'convert':
    # fetch the current arg
    tool_call['args']['conversion_rate'] = conversion_rate #adding a new key–value pair ("conversion_rate": conversion_rate) into that dictionary.
    tool_message2 = convert.invoke(tool_call)
    messages.append(tool_message2)

llm_with_tools.invoke(messages).content #final response after tool execution
