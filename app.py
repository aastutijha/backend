from flask import Flask, jsonify, request
from flask_cors import CORS
from text_summary import textrank_summarizer, get_text_from_link

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

@app.route('/analyze', methods=['POST'])
def analyse():
    data = request.get_json()
    rawtext = data['rawtext']
    
    link = data.get('link', None) # to check link if present
    
    if '://' in rawtext:
        rawtext = get_text_from_link(rawtext)
    
    percentage = int(data.get('percentage', 30) or 30)
    
    result = textrank_summarizer(rawtext, percentage)
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)