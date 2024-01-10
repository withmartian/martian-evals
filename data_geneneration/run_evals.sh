# execute command in root directory
export EVALS_THREADS='20' # number of threads to use

model_names=("gpt-3.5-turbo-1106" "gpt-4-1106-preview")
eval_names=('mmlu-abstract-algebra' 'mmlu-anatomy' 'mmlu-astronomy' 'mmlu-business-ethics' 'mmlu-clinical-knowledge' 'mmlu-college-biology' 'mmlu-college-chemistry' 'mmlu-college-computer-science' 'mmlu-college-mathematics' 'mmlu-college-medicine' 'mmlu-college-physics' 'mmlu-computer-security' 'mmlu-conceptual-physics' 'mmlu-econometrics' 'mmlu-electrical-engineering' 'mmlu-elementary-mathematics' 'mmlu-formal-logic' 'mmlu-global-facts' 'mmlu-high-school-biology' 'mmlu-high-school-chemistry' 'mmlu-high-school-computer-science' 'mmlu-high-school-european-history' 'mmlu-high-school-geography' 'mmlu-high-school-government-and-politics' 'mmlu-high-school-macroeconomics' 'mmlu-high-school-mathematics' 'mmlu-high-school-microeconomics' 'mmlu-high-school-physics' 'mmlu-high-school-psychology' 'mmlu-high-school-statistics' 'mmlu-high-school-us-history' 'mmlu-high-school-world-history' 'mmlu-human-aging' 'mmlu-human-sexuality' 'mmlu-international-law' 'mmlu-jurisprudence' 'mmlu-logical-fallacies' 'mmlu-machine-learning' 'mmlu-management' 'mmlu-marketing' 'mmlu-medical-genetics' 'mmlu-miscellaneous' 'mmlu-moral-disputes' 'mmlu-moral-scenarios' 'mmlu-nutrition' 'mmlu-philosophy' 'mmlu-prehistory' 'mmlu-professional-accounting' 'mmlu-professional-law' 'mmlu-professional-medicine' 'mmlu-professional-psychology' 'mmlu-public-relations' 'mmlu-security-studies' 'mmlu-sociology' 'mmlu-us-foreign-policy' 'mmlu-virology' 'mmlu-world-religions' 'hellaswag')
error_file="~/Desktop/ICML_paper_data_records/failed_param_sets.txt"

# Iterate over model_names and eval_names
for model in "${model_names[@]}"; do
    for eval in "${eval_names[@]}"; do
        echo "Running eval: $eval, model: $model"

        time_string=$(date "+%m-%d_%H:%M:%S")
        start_time=$(date +%s)

        python3 evals/cli/oaieval.py $model $eval --record_path=~/Desktop/ICML_paper_data_records/${time_string}_${model}_$eval.jsonl  > /dev/null 2>&1

        # if error
        if [ $? -ne 0 ]; then
            echo "\nCommand failed:"
            echo "python3 evals/cli/oaieval.py $model $eval --record_path=~/Desktop/ICML_paper_data_records/${time_string}_${model}"
            # If python3 command fails, log the error
            echo "" >> $error_file
            echo "python3 evals/cli/oaieval.py $model $eval --record_path=~/Desktop/ICML_paper_data_records/${time_string}_${model}" >> $error_file
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
