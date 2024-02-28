import os

from dotenv import load_dotenv
load_dotenv()

import assemblyai as aai
aai.settings.api_key = os.environ.get("ASSEMBLY_KEY")

import openai
openai.api_key = os.environ.get("OPEN_AI_API_KEY")

import time
def watcher(func):
	def wrapper(*args, **kwargs):

		# Run and print the time
		time_i = time.time()
		print(f"Starting {func.__name__} ....... ", end="\n")
		result = func(*args, **kwargs)
		print(f"Done in {round(time.time() - time_i, 2)} seconds")

		# Saving the results
		with open(f'temp/{func.__name__}.txt', 'w') as f:
			f.write(str(result))

		# Returning the result
		return result

	return wrapper

# ----> THIS IS THE MAIN FUNCTION FOR TRANSCRIPTION AND DIARIZATION
def transcribe(file_path, num_speakers, lang):
	# transcription
	transcript = transcribe_assembly(file_path, num_speakers, lang)
	# transcript = [('A', "You see me in a ring fighting. It's not just to prove I can beat this man. It's to beat this man and to go back to Chicago and walk skid row, go to Harlem, where the black people are taking needles every day. Dope is a big thing in America. People are dying. Black women are walking the streets, prostitute themselves. When Muhammad Ali come out for 1 hour, they're righteous. So my fighting is for purpose. To pray to Allah, represent Allah, and go back to the hells of America and walk the streets of my downtrodden people. Because all big black people don't no more want to be black. I can say it loudly on television here in the world. I'm the only big million dollar making in the white man's world of prestige, who speaks his people 100% and don't give a damn about the money been shot, take the title, take it all and go to jail tomorrow. I love my people and I'm not going to sell them out, make no movies and mislead them and married blondes, because I'm one negro who the boss let me come in the castle and the rest of them in the fields catching hell. I'm going to the fields and represent them if it means die. So that's why I'll fight.")]

	# speaker recognition aka sr
	sr_prompt_template = sr_get_prompt_template(transcript)
	sr_prompt = sr_get_prompt(sr_prompt_template)
	sr_response = sr_get_response(sr_prompt)
	sr_speakers = sr_get_speakers(sr_response)
	transcript = sr_embed_speakers(transcript, sr_speakers)

	# report
	description = describe(transcript)

	return transcript, description

# --------------------------------------------------

@watcher
def transcribe_assembly(file_path, num_speakers, lang):

	config = aai.TranscriptionConfig(
		speaker_labels=True,
		speakers_expected=num_speakers,
		language_code=lang,
	)

	transcript = aai.Transcriber().transcribe(file_path, config)

	return [(utt.speaker, utt.text) for utt in transcript.utterances]

# --------------------------------------------------

@watcher
def sr_get_prompt_template(transcript):
	conversation = transcript[:10]
	unique_speakers = sorted(list(set([i[0] for i in conversation])))
	speakers = {
		k: "<SPEAKER NAME>" for k in unique_speakers
	}
	return conversation, speakers

@watcher
def sr_get_prompt(sr_promp_template):
	strating_conversation, speakers_template = sr_promp_template
	return f"Given the following conversation:\n\n{strating_conversation}\n\nInterpret the speaker names and complete the speaker dictionary like in the template:\n\n{speakers_template}\n\nReturn only the speaker dictionary nothing else."


@watcher
def sr_get_response(sr_prompt):
	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo",
		messages=[{"role": "user", "content": sr_prompt}],
		temperature=0,
	)
	return response.choices[0].message['content']

@watcher
def sr_get_speakers(sr_response):
	try:
		speakers = eval(sr_response)
		# response_de = eval(sr_response)
		# if type(response_de) == list:
		# 	response_de = response_de[0]
		# speakers = response_de.get("speakers", {})
	except Exception as e:
		print(e)
		speakers = {}
	return speakers

@watcher
def sr_embed_speakers(transcript, speakers):
	final_transcription = []
	for speaker, said in transcript:
		speaker = speakers.get(
			speaker,
			f"Unidentified"
		)
		final_transcription.append(
			f"{speaker}: {said}"
		)
	final_transcription = "\n".join(final_transcription)
	return final_transcription

# --------------------------------------------------

@watcher
def describe(transcript):

	content = f"Following is the conversation, generate me Main point of in bullet conversation and on second heading create the detailed summary of conversation : {transcript}"

	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo",
		messages=[{"role": "user", "content": content}],
		temperature=0.2,
	)

	description = response.choices[0].message["content"]

	return description