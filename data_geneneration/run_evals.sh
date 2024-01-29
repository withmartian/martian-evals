# execute command in root directory
eval_names=('hellaswag' 'arc-challenge' 'arc-challenge-test' 'arc-challenge-val' 'winogrande' 'mbpp' 'grade-school-math' 'mtbench' 'mtbench-reference' 'mtbench-math' 'mmlu-abstract-algebra' 'mmlu-anatomy' 'mmlu-astronomy' 'mmlu-business-ethics' 'mmlu-clinical-knowledge' 'mmlu-college-biology' 'mmlu-college-chemistry' 'mmlu-college-computer-science' 'mmlu-college-mathematics' 'mmlu-college-medicine' 'mmlu-college-physics' 'mmlu-computer-security' 'mmlu-conceptual-physics' 'mmlu-econometrics' 'mmlu-electrical-engineering' 'mmlu-elementary-mathematics' 'mmlu-formal-logic' 'mmlu-global-facts' 'mmlu-high-school-biology' 'mmlu-high-school-chemistry' 'mmlu-high-school-computer-science' 'mmlu-high-school-european-history' 'mmlu-high-school-geography' 'mmlu-high-school-government-and-politics' 'mmlu-high-school-macroeconomics' 'mmlu-high-school-mathematics' 'mmlu-high-school-microeconomics' 'mmlu-high-school-physics' 'mmlu-high-school-psychology' 'mmlu-high-school-statistics' 'mmlu-high-school-us-history' 'mmlu-high-school-world-history' 'mmlu-human-aging' 'mmlu-human-sexuality' 'mmlu-international-law' 'mmlu-jurisprudence' 'mmlu-logical-fallacies' 'mmlu-machine-learning' 'mmlu-management' 'mmlu-marketing' 'mmlu-medical-genetics' 'mmlu-miscellaneous' 'mmlu-moral-disputes' 'mmlu-moral-scenarios' 'mmlu-nutrition' 'mmlu-philosophy' 'mmlu-prehistory' 'mmlu-professional-accounting' 'mmlu-professional-law' 'mmlu-professional-medicine' 'mmlu-professional-psychology' 'mmlu-public-relations' 'mmlu-security-studies' 'mmlu-sociology' 'mmlu-us-foreign-policy' 'mmlu-virology' 'mmlu-world-religions')
openai_model_names=("gpt-3.5-turbo-1106" "gpt-4-1106-preview")
together_model_names=("zero-one-ai/Yi-34B-Chat" "WizardLM/WizardLM-13B-V1.2")
model_names=('claude-v1' 'claude-v2' "claude-instant-v1" "meta/code-llama-instruct-34b-chat" "meta/llama-2-70b-chat" "mistralai/mixtral-8x7b-chat" "mistralai/mistral-7b-chat")
error_file="./failed_param_sets.txt"
data_path="./"

# Iterate over model_names and eval_names
EVALS_THREADS='16' # number of threads to use
for model in "${openai_model_names[@]}"; do
  for eval in "${eval_names[@]}"; do
        echo "\n\n Running eval: $eval, model: $model"

        time_string=$(date "+%m-%d_%H:%M:%S")
        start_time=$(date +%s)
        EVALS_THREAD_TIMEOUT=600 OPENAI_API_KEY=$REAL_OPENAI_API_KEY oaieval $model $eval --record_path=${data_path}/${model}__$eval.jsonl #  > /dev/null 2>&1
        # if error
        if [ $? -ne 0 ]; then
            echo "\nCommand failed:"
            echo "python3 evals/cli/oaieval.py $model $eval --record_path=${data_path}/${time_string}__${model}__$eval.jsonl"
            # If python3 command fails, log the error
            echo "" >> $error_file
            echo "python3 evals/cli/oaieval.py $model $eval --record_path=${data_path}/${time_string}__${model}__$eval.jsonl" >> $error_file
            # Skip to the next iteration
            continue
        fi

        # Calculate and print the duration
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "Duration for eval_name $eval: $duration seconds"
        echo ""

    done
done

EVALS_THREADS='16' # number of threads to use
for model in "${model_names[@]}"; do
  for eval in "${eval_names[@]}"; do
        echo "\n\n Running eval: $eval, model: $model"

        time_string=$(date "+%m-%d_%H:%M:%S")
        start_time=$(date +%s)
        EVALS_THREAD_TIMEOUT=600 NUMEXPR_MAX_THREADS=16 OPENAI_API_KEY=$MARTIAN_API_KEY OPENAI_URL=https://staging.withmartian.com/api/openai/v1 oaieval $model $eval --record_path=${data_path}/${model}__$eval.jsonl #  > /dev/null 2>&1
        # if error
        if [ $? -ne 0 ]; then
            echo "\nCommand failed:"
            echo "python3 evals/cli/oaieval.py $model $eval --record_path=${data_path}/${time_string}__${model}__$eval.jsonl"
            # If python3 command fails, log the error
            echo "" >> $error_file
            echo "python3 evals/cli/oaieval.py $model $eval --record_path=${data_path}/${time_string}__${model}__$eval.jsonl" >> $error_file
            # Skip to the next iteration
            continue
        fi

        # Calculate and print the duration
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "Duration for eval_name $eval: $duration seconds"
        echo ""

    done
done

EVALS_THREADS='16' # number of threads to use
for model in "${together_model_names[@]}"; do
  for eval in "${eval_names[@]}"; do
        echo "\n\n Running eval: $eval, model: $model"

        time_string=$(date "+%m-%d_%H:%M:%S")
        start_time=$(date +%s)
        EVALS_THREAD_TIMEOUT=600 NUMEXPR_MAX_THREADS=8 OPENAI_API_KEY=$TOGETHER_API_KEY OPENAI_URL=https://api.together.xyz oaieval $model $eval --record_path=${data_path}/${model}__$eval.jsonl #  > /dev/null 2>&1
        # if error
        if [ $? -ne 0 ]; then
            echo "\nCommand failed:"
            echo "python3 evals/cli/oaieval.py $model $eval --record_path=${data_path}/${time_string}__${model}__$eval.jsonl"
            # If python3 command fails, log the error
            echo "" >> $error_file
            echo "python3 evals/cli/oaieval.py $model $eval --record_path=${data_path}/${time_string}__${model}__$eval.jsonl" >> $error_file
            # Skip to the next iteration
            continue
        fi

        # Calculate and print the duration
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "Duration for eval_name $eval: $duration seconds"
        echo ""

    done
done