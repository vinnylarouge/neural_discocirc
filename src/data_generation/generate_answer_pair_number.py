from discopy import Ty


def find_wire(diagram, name):
    index = 0
    box = diagram.boxes[index]
    while box.dom == Ty():
        if box.name == name:
            return diagram.offsets[index]

        index += 1
        box = diagram.boxes[index]

    raise Exception('wire not found')


def get_qa_numbers(context_circuits, questions, answers):
    """
    Parameters:
    -----------
    context_circuits : list
        List of context circuits.
    questions : list
        List of strings
    answers : list
        List of strings
    
    Returns:
    --------
    q_a_number_pairs : list
        List of tuples (question word index, answer word index)
    """
    q_a_pairs = []
    # the following is quite hard-coded
    for question, answer_word in zip(questions, answers):
        # the 'question word' for tasks 1, 2 is always the last word in the question
        question_word = question.split()[-1][:-1]
        q_a_pairs.append((question_word, answer_word))

    q_a_number_pairs = []
    for i, (question, answer) in enumerate(q_a_pairs):
        q_id = find_wire(context_circuits[i], question)
        a_id = find_wire(context_circuits[i], answer)
        q_a_number_pairs.append((q_id, a_id))

    return q_a_number_pairs
