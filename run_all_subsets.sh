logfile="dpr_results_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$logfile") 2>&1
echo "Logging to $logfile"

splits=("train" "dev1" "dev2" "dev3")
subsets=("train100k" "eval250k")
runfiles=("pretrained" "finetuned" "finetuned_new")

metrics="ndcg@10 ndcg@1000 R@1000 rr map map@1000 P@10 R@10 f1@10 success@10"

# Pretrained and finetuned
for runf in "${runfiles[@]}"; do
    # Train and Eval
    for subset in "${subsets[@]}"; do
        if [[ "$subset" == "train100k" ]]; then
            folder="train80"
        elif [[ "$subset" == "eval250k" ]]; then
            folder="eval20"
        else
            echo "Unknown subset: $subset"
            continue
        fi
        # train, dev1, dev2, dev3
        for split in "${splits[@]}"; do
            run="DPR/runs/${runf}/${folder}/${split}.txt"

            echo
            echo "=============================="
            echo "Running embeddings=$runf split=$split subset=$subset"
            echo "Run file: $run"
            echo "=============================="
            
            python EvalPipelineSubSet/run_eval.py \
                --split "$split" \
                --run "$run" \
                --metrics $metrics \
                --subset "$subset"
        done
    done
done