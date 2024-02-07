import streamlit as st
from google.cloud import language
import vertexai
import json
import requests
from vertexai.language_models import TextGenerationModel

# Initialize Vertex AI
vertexai.init(project="sascha-playground-doit", location="us-central1")
model = TextGenerationModel.from_pretrained("text-bison")

PROJECT_ID = "sascha-playground-doit"
MODEL_ID = "text-bison"
GOOGLE_MODERATION_API_COST_PER_100_CHAR = 0.0005
TOXIC_THRESHOLD = 0.20  # 20%

def calculate_google_moderation_cost(text: str) -> float:
    return (len(text) / 100) * GOOGLE_MODERATION_API_COST_PER_100_CHAR

def calculate_gen_ai_cost(text: str) -> float:
    return (len(text) / 1000) * 0.0005


def is_content_toxic(categories):
    for category in categories:
        if category.name == "Toxic" and category.confidence > TOXIC_THRESHOLD:
            return True
    return False

def display_toxicity(response):
    if response.get("toxic", False):
        st.markdown(f"### :warning: Toxic Content Detected")
        st.markdown(f"**Reason:** {response.get('reason', 'Not specified')}")
        st.markdown("This content has been flagged as toxic and may require further action.")
    else:
        st.markdown("### :white_check_mark: Content is Non-Toxic")


def moderate_text(text: str) -> language.ModerateTextResponse:
    client = language.LanguageServiceClient()
    document = language.Document(
        content=text,
        type_=language.Document.Type.PLAIN_TEXT,
    )
    return client.moderate_text(document=document)

def get_llm_response(text: str) -> str:
    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 40
    }
    
    json_string = '{"reason": "", "toxic": false}'

    
    prompt = f"""You are a experienced content moderation system classify if content is toxic.
    Content could be also sarcastic and still be toxic keep that in mind.
    
    Return the result as valid JSON in the following format:
    {json_string}
    
    Content:
    {text}
    """
    response = model.predict(prompt, **parameters)
    return response.text

def main():
    
    st.set_page_config(layout="wide")
    st.title("Content Moderation Comparison")

    user_input = st.text_area("Enter text to analyze", "")
   
    analyze = st.button("Analyze")
    
    col1, col2 = st.columns(2)
    
    if analyze:
    
            with col1:
                st.title("Google Moderation API")
                st.subheader("Costs")
                google_cost = calculate_google_moderation_cost(user_input)
                st.text(f"${google_cost:.10f}")
                st.text("First 50K requests per month are free")
            
                
                st.subheader("Results")
                
                
            with col2:
                st.title("Gen AI Custom Solution")
                st.subheader("Costs")
                gen_ai_cost = calculate_gen_ai_cost(user_input)
                st.text(f"${gen_ai_cost:.10f}")
            
                st.subheader("Results")
        
            try:
                    

                    with col1:
                        with st.spinner("Analyzing..."):
                            mod_response = moderate_text(user_input)
                            if mod_response.moderation_categories:
                                
                                toxic = is_content_toxic(mod_response.moderation_categories)
                                if toxic:
                                    st.error("Content is not safe :x:")
                                else:
                                    st.success("Content is safe :white_check_mark:")
                                
                                categories = [{"Category": cat.name, "Confidence": cat.confidence} 
                                            for cat in mod_response.moderation_categories]
                                st.table(categories)
                                
                               
                                
                            else:
                                st.write("No moderation categories found.")
                            

                    with col2:
                        with st.spinner("Analyzing..."):
                            llm_response_str = get_llm_response(user_input)
                            
                            
                            start = llm_response_str.find('{')
                            end = llm_response_str.rfind('}') + 1
                            json_str = llm_response_str[start:end]
                            
                            llm_response_str = json.loads(json_str)  # Parse JSON string
                           

                            if llm_response_str.get("toxic", False):
                                st.error("Content is not safe :x: - " + llm_response_str.get("reason", ""))
                            else:
                                st.success("Content is safe :white_check_mark:")

                            st.json(llm_response_str)  # Display the parsed JSON
                           
                            #llm_response_json = json.loads(llm_response)
                            #display_toxicity(llm_response_json, col2)

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()