# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from vllm import LLM, SamplingParams

from minference import MInference  # including MInference

import os

def main() -> None:
    assert os.environ.get('VLLM_USE_V1') == '0', f'using v1, VLLM_USE_V1: {os.environ.get("VLLM_USE_V1")}'
    assert os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING') == '0', f'using multiprocessing, VLLM_ENABLE_V1_MULTIPROCESSING: {os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")}'

    prompts = [
        # "Hello my name is",    # * 4 words
        # "The president of the United States is",    # * 7 words
        # "The capital of France is cold",    # * 6 words
        # "The future of AI is",      # * 5 words
        "Shakespeare was born and raised in Stratford-upon-Avon, Warwickshire. At the age of 18, he married Anne Hathaway, with whom he had three children: Susanna, and twins Hamnet and Judith. Sometime between 1585 and 1592 he began a successful career in London as an actor, writer, and part-owner (\"sharer\") of a playing company called the Lord Chamberlain's Men, later known as the King's Men after the ascension of King James VI of Scotland to the English throne. At age 49 (around 1613) he appears to have retired to Stratford, where he died three years later. Few records of Shakespeare's private life survive; this has stimulated considerable speculation about such matters as his physical appearance, his sexuality, his religious beliefs and even certain fringe theories as to whether the works attributed to him were written by others. Shakespeare was born and raised in Stratford-upon-Avon, Warwickshire. At the age of 18, he married Anne Hathaway, with whom he had three children: Susanna, and twins Hamnet and Judith. Sometime between 1585 and 1592 he began a successful career in London as an actor, writer, and part-owner (\"sharer\") of a playing company called the Lord Chamberlain's Men, later known as the King's Men after the ascension of King James VI of Scotland to the English throne. At age 49 (around 1613) he appears to have retired to Stratford, where he died three years later. Few records of Shakespeare's private life survive; this has stimulated considerable speculation about such matters as his physical appearance, his sexuality, his religious beliefs and even certain fringe theories as to whether the works attributed to him were written by others. Shakespeare was born and raised in Stratford-upon-Avon, Warwickshire. At the age of 18, he married Anne Hathaway, with whom he had three children: Susanna, and twins Hamnet and Judith. Sometime between 1585 and 1592 he began a successful career in London as an actor, writer, and part-owner ",
#         """
# From fairest creatures we desire increase,
# That thereby beauty's rose might never die,
# But as the riper should by time decease,
# His tender heir might bear his memory:
# But thou contracted to thine own bright eyes,
# Feed'st thy light's flame with self-substantial fuel,
# Making a famine where abundance lies,
# Thy self thy foe, to thy sweet self too cruel:
# Thou that art now the world's fresh ornament,
# And only herald to the gaudy spring,
# Within thine own bud buriest thy content,
# And, tender churl, mak'st waste in niggarding:
# Pity the world, or else this glutton be,
# To eat the world's due, by the grave and thee.
#
# When forty winters shall besiege thy brow,
# And dig deep trenches in thy beauty's field,
# Thy youth's proud livery so gazed on now,
# Will be a totter'd weed of small worth held:
# Then being asked, where all thy beauty lies,
# Where all the treasure of thy lusty days;
# To say, within thine own deep sunken eyes,
# Were an all-eating shame, and thriftless praise.
# How much more praise deserv'd thy beauty's use,
# If thou couldst answer 'This fair child of mine
# Shall sum my count, and make my old excuse,'
# Proving his beauty by succession thine!
# This were to be new made when thou art old,
# And see thy blood warm when thou feel'st it cold.
#
# Look in thy glass and tell the face thou viewest
# Now is the time that face should form another;
# Whose fresh repair if now thou not renewest,
# Thou dost beguile the world, unbless some mother.
# For where is she so fair whose uneared womb
# Disdains the tillage of thy husbandry?
# Or who is he so fond will be the tomb
# Of his self-love, to stop posterity?
# Thou art thy mother's glass and she in thee
# Calls back the lovely April of her prime;
# So thou through windows of thine age shalt see,
# Despite of wrinkles, this thy golden time.
# But if thou live, remembered not to be,
# Die single and thine image dies with thee."""

        # "Shakespeare was born and raised in Stratford upon Avon Warwickshire At the age of 18 he Shakespeare was born and raised in Stratford upon Avon Warwickshire At the age of 18 he"
    ]

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        max_tokens=30,
        seed=42
    )

    model_name = "./models--Qwen--Qwen2-0.5B"
    
    llm = LLM(
        model=model_name,
        max_num_seqs=1,
        enforce_eager=True,     # disable to get 2-3x faster speed for CUDA graph
        dtype='float16',
        max_model_len=12800,
        block_size=256,
        # block_size=16,
        # enable_chunked_prefill=True,
        # max_num_batched_tokens=64
        # max_num_batched_tokens=16
    )

    # Patch MInference Module
    minference_patch = MInference("vllm_minference", model_name)
    llm = minference_patch(llm)

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == '__main__':
    main()
