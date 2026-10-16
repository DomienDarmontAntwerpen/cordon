"""
The request pipeline, one file per step, in order:

    step1_admission         may it come in? (key, model, size, rate, budget)
    step2_input_protection  mask personal data, detect prompt injection
    step3_loop_detection    stop agents stuck in a loop
    step4_optimize          shrink: duplicates, long history, tool output, images
    step5_cache             answer from the cache
    step6_routing           choose the model(s)
    step7_generate          call the model, fail over down the ladder
    step8_quality           retry a bad cheap answer once with the strong model
    step9_output            protect the answer, restore values, book, reply

Start reading at run.py. context.py holds the request's state; streaming.py
is steps 7-9 for streamed answers; passthrough.py handles an unexpected crash.
"""
