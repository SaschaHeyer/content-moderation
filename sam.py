from google.cloud import language

def moderate_text(text: str) -> language.ModerateTextResponse:
    client = language.LanguageServiceClient()
    document = language.Document(
        content=text,
        type_=language.Document.Type.PLAIN_TEXT,
    )
    return client.moderate_text(document=document)

text = (
    "you are a piece of ****"
)
response = moderate_text(text)
print(response)