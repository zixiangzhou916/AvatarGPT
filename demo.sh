clear

../../dockers/UDE-env/bin/python3.8 scripts/evaluate.py \
    --task AvatarGPTEvaluator \
    --config configs/llm_t5/config_t5_large.yaml \
    --eval_folder demo_outputs \
    --eval_name demo \
    --repeat_times 1 \
    --eval_mode se1 \
    --eval_task t2m,m2m,m2t,ct2t,cs2s,ct2s,cs2t,t2c,s2c,t2s,s2t \
    --eval_pipeline p0 \
    --topk 1 \
    --use_semantic_sampling True \
    --temperature 1.0 \
    --demo_list demo_inputs/test.txt \
    --demo_data demo_inputs/samples

../../dockers/SelfRecon/bin/python3.8 render/convert_hml_to_mesh.py \
    --filedir demo_outputs/demo/output/se1_p0/ \
    --mesh_dir demo_outputs/demo/meshes/se1_p0/ \
    --video_dir demo_outputs/demo/videos/se1_p0/ \
    --visualize False \
    --convert_gt False

PYOPENGL_PLATFORM=osmesa /opt/conda/bin/python3.8 render/mesh_renderer.py \
    --mesh_dir demo_outputs/demo/meshes/se1_p0/ \
    --video_dir demo_outputs/demo/videos/se1_p0/ \
    --run_mode dynamic \
    --sample_rate 4 \
    --num_frame 20