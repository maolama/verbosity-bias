import json
import os


def directory_analyze_experiment_consistency(input_dir):
    all_filenames = [f for f in os.listdir(input_dir)]
    all_filenames.sort()
    for filename in all_filenames:
        if filename.endswith('.json'):
            question_file = os.path.join(input_dir, filename)
            with open(question_file, 'r', encoding='utf-8') as qf:
                question = json.load(qf)
            question = analyze_experiment_consistency(question)
            with open(question_file, 'w', encoding='utf-8') as output_file:
                json.dump(question, output_file, ensure_ascii=False, indent=4)
    pass


def analyze_experiment_consistency(data):
    """
    Analyzes experiment data for consistency between direct and reverse runs.

    Args:
        data: A dictionary parsed from the JSON object containing the experiments.

    Returns:
        A dictionary mapping each experiment ID to its consistency status.
    """
    if 'experiments' not in data:
        print("Error: 'experiments' key not found in the provided data.")
        return {}

    experiments = data['experiments']
    consistency_results = {}

    for exp_id, details in experiments.items():
        try:
            direct_choice = details['direct']['extracted_answer']
            reverse_choice = details['reverse']['extracted_answer']

            status = ""

            # Rule 1: Check for perfect consistency
            if (direct_choice == 'A' and reverse_choice == 'B') or \
                    (direct_choice == 'B' and reverse_choice == 'A') or \
                    (direct_choice == 'C' and reverse_choice == 'C'):
                status = "Consistent"

            # Rule 2: Check for clear position bias
            elif (direct_choice == 'A' and reverse_choice == 'A') or \
                    (direct_choice == 'B' and reverse_choice == 'B'):
                status = "Inconsistent (Position Bias)"
                print(status)

            # Rule 3: All other cases are unstable evaluations
            else:
                status = "Inconsistent (Unstable Evaluation)"
                print(status)

            consistency_results[exp_id] = {
                "status": status,
                "direct": direct_choice,
                "reverse": reverse_choice
            }
            details['status'] = status
            details['result'] = direct_choice if status == 'Consistent' else None

        except KeyError as e:
            # Handle cases where the experiment structure is malformed
            consistency_results[exp_id] = {
                "status": f"Error: Malformed data - missing key {e}",
                "direct": "N/A",
                "reverse": "N/A"
            }

    return data


# --- Example Usage ---
if __name__ == "__main__":
    base_dir = '../../data/evaluation'
    models = [m for m in os.listdir(base_dir)]
    models.sort()
    for model in models:
        categories = [g for g in os.listdir(os.path.join(base_dir, model))]
        categories.sort()
        for category in categories:
            dir_add = os.path.join(base_dir, model, category)
            print(dir_add)
            directory_analyze_experiment_consistency(dir_add)
    # dir = '..\..\data\evaluation\deepseek-r1\Algorithmic'
    # directory_analyze_experiment_consistency(dir)
