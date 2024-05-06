Make sure you have an active ChatGPT API from [OpenAI](https://openai.com/). 

We assume the description of video content is already prepared and placed at:

    data_curation_pipeline/sample_videos/desc

Now you can generate dataset to train ***AvatarGPT*** by running following:

    cd data_curation_pipeline/ChatGPT

    INPUT_DIR=../sample_videos/desc
    OUTPUT_DIR=../sample_videos/final

    python run_chatgpt.py --task detail --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR}

The generated dataset will be placed at:

    data_curation_pipeline/sample_videos/final