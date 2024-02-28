from flask import Flask, render_template, request, jsonify
from audio import transcribe

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/handle_audio_file', methods=['POST'])
def handle_audio_file():

    print("Hit at handle_audio_file")
    try:
        audio_file = request.files['audioFile']
        num_speakers = request.form.get('n_speakers')
        lang = request.form.get('lang')

        if audio_file and (audio_file.mimetype.startswith('audio/mp3') or audio_file.mimetype.startswith('audio/wav') or audio_file.filename.endswith('.mp3') or audio_file.filename.endswith('.wav')) or audio_file.filename.endswith('.m4a'):

            file_path = 'uploads/' + audio_file.filename
            audio_file.save(file_path)
            transcription, description = transcribe(file_path, num_speakers, lang)

            return jsonify({
                'message': 'File uploaded successfully',
                'transcription': transcription,
                'description': description,
            })

        else:
            return jsonify({'error': 'Selected file is not an audio file or has an unsupported format'})

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
