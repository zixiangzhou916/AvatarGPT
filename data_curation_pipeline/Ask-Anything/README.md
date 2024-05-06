Clone [Ask-Anything](https://github.com/OpenGVLab/Ask-Anything) and put the ***video_chat***  folder here.

Please follow the [instruction](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat) to setup the enviroment. We use the 7B model, you can download it from [here](https://drive.google.com/file/d/1C4s65TC5Zr85I8dZmnfrrw6oDAjj1H4P/view).

We provide a sample video to demonstrate the pipeline of auto-data curation, the video is placed at:
    
    data_curation_pipeline/Ask-Anything/sample_videos/raw

To obtain detailed textural description of the video, you can run the following command:

    cd data_curation_pipeline/Ask-Anything/video_chat
    
    DIRECTORY_TO_THE_VIDEOS=../sample_videos/raw
    DIRECTORY_TO_THE_VIDEO_DESCRIPTIONS=../sample_videos/desc

    python run_ask_anything.py --task detail --input_video_dir ${DIRECTORY_TO_THE_VIDEOS} --output_video_dir ${DIRECTORY_TO_THE_VIDEO_DESCRIPTIONS} --max_new_tokens 200

The detail description of the video content organized in **.json** files will be placed at:
    
    data_curation_pipeline/Ask-Anything/sample_videos/desc

