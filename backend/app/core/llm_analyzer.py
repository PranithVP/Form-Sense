import os
import asyncio
from together import Together
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Together.ai client
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=TOGETHER_API_KEY)

async def analyze_form_with_llm(exercise_data: Dict[str, Any]) -> str:
    """
    Analyze exercise form using Together.ai's LLM.
    
    Args:
        exercise_data (Dict[str, Any]): Dictionary containing exercise data and analysis
        
    Returns:
        str: LLM-generated feedback about the exercise form
    """
    # Extract exercise name and summary
    exercise_name = exercise_data.get("classified_exercise", "Unknown Exercise")
    angle_analysis = exercise_data.get("angle_analysis", {})
    summary = angle_analysis.get("summary", "No summary available")
    
    # Construct a simpler prompt, asking for markdown format without numbered headings
    prompt = f"""You are an expert fitness trainer and form coach. Based on the following exercise data, provide detailed feedback about the user's form for {exercise_name}.
Exercise Data:
{summary}

Please provide detailed feedback about their form. Use headings and bullet points. Do NOT use markdown formatting (no ** for bold text, no # for headings). Do NOT number the main sections.

Overall Assessment of Form

Specific Areas that Need Improvement
- 
- 

Suggestions for Correction
- 
- 

Safety Concerns
- 

Tips for Better Performance
- 
- """
    
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.chat.completions.create,
                model="mistralai/Mistral-7B-Instruct-v0.2",
                messages=[{"role": "user", "content": prompt}]
            ),
            timeout=45
        )
        
        return response.choices[0].message.content.strip()
        
    except asyncio.TimeoutError:
        return "Error: Request timed out while generating form analysis."
    except Exception as e:
        return f"Error generating form analysis: {str(e)}" 