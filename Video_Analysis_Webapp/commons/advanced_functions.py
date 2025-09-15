import os
from datetime import datetime
from model.class_video_copilot import VideoToSpeechClass, VideoToObjectsClass, VideoTopicsSummaryClass, VideoPostValidationClass
from verifications.check_policy_compliance import *
from verifications.build_analysis_json import *
from model.constants import *


def process_video_speech(video_path):
    """Extract and process speech from video"""
    videoToSpeech = VideoToSpeechClass(video_path)
    text_video = videoToSpeech.extract_speech_google_recognizer(TEMP_AUDIO_FILE)
    text_video = videoToSpeech.process_text()
    
    videoToSpeech.save_speech_from_video(video_path, text_video)
    
    return {
        'text_video': text_video,
        'videoToSpeech': videoToSpeech,
        'summary_text': '',
        'text_NER': '',
        'text_NER_surrounding': ''
    }


def perform_text_analysis(analysis_results, filename):
    """Perform comprehensive text analysis"""
    text_video = analysis_results['text_video']
    videoTopicsSummary = VideoTopicsSummaryClass(text_video, ['test'], 'test.json')
    
    # Generate summary and analysis
    summary_text = videoTopicsSummary.from_text_to_sentences_and_summary()
    text_NER = str(videoTopicsSummary.perform_ner_analysis_second())
    text_NER_surrounding = str(videoTopicsSummary.get_entities_surrounding_infos(window_size=5))
    
    # Extract key information and perform sentiment analysis
    videoTopicsSummary.extract_key_infos()
    
    current_time = datetime.now().strftime('%H-%M-%S')
    json_sentiments_filename = f"{os.path.splitext(filename)[0]}_{current_time}_json_sentiments_per_sentence.json"
    videoTopicsSummary.sentiment_analysis_per_summary_sentence(json_sentiments_filename)
    
    # Save analysis to JSON
    analysis_json_filename = f"{os.path.splitext(filename)[0]}_{current_time}_json_video_analysis.json"
    json_file = read_fill_save_json_file(analysis_json_filename, analysis_results['videoToSpeech'].video_path, 
                                        text_video, videoTopicsSummary.entities, 
                                        videoTopicsSummary.key_infos, videoTopicsSummary.sentiment_scores)
    
    # Update results
    analysis_results.update({
        'summary_text': summary_text,
        'text_NER': text_NER,
        'text_NER_surrounding': text_NER_surrounding,
        'videoTopicsSummary': videoTopicsSummary,
        'json_file': json_file
    })


def check_compliance(analysis_results, filename):
    """Check policy compliance"""
    conditions_met = (analysis_results['text_video'] and 
                     analysis_results['summary_text'] and 
                     analysis_results['summary_text'] != analysis_results['videoTopicsSummary'].default_text)
    
    if conditions_met and analysis_results.get('json_file'):
        policy_data = read_json_file(POLICY_DATA_FILE)
        analysis_data = read_json_file(analysis_results['json_file'])
        
        if policy_data and analysis_data:
            compliance_dict, compliance_metrics = check_policy_compliance(policy_data, analysis_data)
        else:
            compliance_dict, _ = check_policy_compliance_default()
    else:
        compliance_dict, _ = check_policy_compliance_default()
    
    analysis_results['compliance_dict'] = compliance_dict


def validate_speech(analysis_results, video_path):
    """Validate if speech is AI-generated and extract favorite products"""
    output_text = analysis_results['videoToSpeech'].output_text
    
    if not output_text:
        return
    
    videoPostValidation = VideoPostValidationClass(video_path, output_text)
    
    # Check AI generation
    mode_test_or_production = 'test'
    generated_speech = AI_GENERATED_SPEECH if mode_test_or_production == 'test' else videoPostValidation.get_ai_generated_speech()
    
    similarity_ratio, is_ai_generated = videoPostValidation.is_ai_generated_speech(generated_speech)
    videoPostValidation.save_ai_validation_result(generated_speech, output_text, 
                                                 similarity_ratio, is_ai_generated, video_path)
    
    # Extract favorite products
    videoPostValidation.fetch_count_favorite_products_from_speech()
