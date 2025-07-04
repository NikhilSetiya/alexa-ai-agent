# api/alexa.py (Vercel Serverless Function)
import json
import os
from typing import Dict, Any
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def handler(request):
    """
    Main Vercel serverless function handler
    Processes Alexa requests and returns responses
    """
    try:
        # Parse the request
        if request.method != 'POST':
            return {
                'statusCode': 405,
                'body': json.dumps({'error': 'Method not allowed'})
            }
        
        # Get the request body
        alexa_request = json.loads(request.body) if hasattr(request, 'body') else request.json
        
        # Verify this is an Alexa request
        if not alexa_request.get('request'):
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid Alexa request'})
            }
        
        # Process the request
        response = process_alexa_request(alexa_request)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
            },
            'body': json.dumps(response)
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'version': '1.0',
                'response': {
                    'outputSpeech': {
                        'type': 'PlainText',
                        'text': 'Sorry, I encountered an error. Please try again.'
                    },
                    'shouldEndSession': True
                }
            })
        }

def process_alexa_request(alexa_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process an Alexa request and return appropriate response
    """
    request_type = alexa_request['request']['type']
    
    if request_type == 'LaunchRequest':
        return handle_launch_request()
    elif request_type == 'IntentRequest':
        return handle_intent_request(alexa_request)
    elif request_type == 'SessionEndedRequest':
        return handle_session_ended_request()
    else:
        return create_error_response("Unknown request type")

def handle_launch_request() -> Dict[str, Any]:
    """
    Handle when user opens the skill without a specific intent
    """
    speech_text = "Hello! I'm your AI assistant. How can I help you today?"
    
    return {
        'version': '1.0',
        'response': {
            'outputSpeech': {
                'type': 'PlainText',
                'text': speech_text
            },
            'reprompt': {
                'outputSpeech': {
                    'type': 'PlainText',
                    'text': 'What would you like me to help you with?'
                }
            },
            'shouldEndSession': False
        }
    }

def handle_intent_request(alexa_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle specific intent requests
    """
    intent = alexa_request['request']['intent']
    intent_name = intent['name']
    
    if intent_name == 'ChatIntent':
        return handle_chat_intent(alexa_request)
    elif intent_name == 'AMAZON.HelpIntent':
        return handle_help_intent()
    elif intent_name == 'AMAZON.CancelIntent' or intent_name == 'AMAZON.StopIntent':
        return handle_stop_intent()
    else:
        return create_error_response("I didn't understand that request.")

def handle_chat_intent(alexa_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle the main chat intent with OpenAI
    """
    try:
        # Extract user query
        slots = alexa_request['request']['intent'].get('slots', {})
        user_query = slots.get('query', {}).get('value', '')
        
        if not user_query:
            return create_error_response("I didn't catch what you said. Could you repeat that?")
        
        # Get session info for context
        session = alexa_request.get('session', {})
        user_id = session.get('user', {}).get('userId', 'unknown')
        
        # Generate response using OpenAI
        ai_response = generate_ai_response(user_query, user_id)
        
        return {
            'version': '1.0',
            'response': {
                'outputSpeech': {
                    'type': 'PlainText',
                    'text': ai_response
                },
                'shouldEndSession': False
            }
        }
        
    except Exception as e:
        logger.error(f"Error in chat intent: {str(e)}")
        return create_error_response("Sorry, I had trouble processing your request.")

def generate_ai_response(user_query: str, user_id: str) -> str:
    """
    Generate AI response using OpenAI
    """
    try:
        # System prompt optimized for voice interaction
        system_prompt = """You are a helpful AI assistant integrated with Alexa. 
        
        Important guidelines:
        - Keep responses concise and conversational (2-3 sentences max)
        - Speak naturally as if talking to someone
        - Avoid long lists or complex formatting
        - Be helpful and friendly
        - If you need to provide multiple items, speak them naturally
        - Don't use phrases like "Here's what I found" - just give the information
        
        The user is speaking to you through Alexa, so respond as if you're having a conversation."""
        
        # Create the chat completion
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cost-effective model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            max_tokens=150,  # Keep responses short for voice
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        # Ensure response is appropriate length for Alexa
        if len(ai_response) > 8000:  # Alexa limit
            ai_response = ai_response[:7900] + "..."
        
        return ai_response
        
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return "I'm having trouble connecting to my knowledge base right now. Please try again in a moment."

def handle_help_intent() -> Dict[str, Any]:
    """
    Handle help request
    """
    help_text = "I'm your AI assistant. You can ask me questions, get information, or have a conversation. Just say what you need help with!"
    
    return {
        'version': '1.0',
        'response': {
            'outputSpeech': {
                'type': 'PlainText',
                'text': help_text
            },
            'shouldEndSession': False
        }
    }

def handle_stop_intent() -> Dict[str, Any]:
    """
    Handle stop/cancel request
    """
    return {
        'version': '1.0',
        'response': {
            'outputSpeech': {
                'type': 'PlainText',
                'text': 'Goodbye!'
            },
            'shouldEndSession': True
        }
    }

def handle_session_ended_request() -> Dict[str, Any]:
    """
    Handle session ended (no response needed)
    """
    return {
        'version': '1.0',
        'response': {}
    }

def create_error_response(message: str) -> Dict[str, Any]:
    """
    Create a standardized error response
    """
    return {
        'version': '1.0',
        'response': {
            'outputSpeech': {
                'type': 'PlainText',
                'text': message
            },
            'shouldEndSession': False
        }
    }