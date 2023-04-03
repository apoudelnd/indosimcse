indobertweet_graph_run1 -- first image generated using indobertweet
run on fine-tuned model as well

model folder is for representing the sentences into 2 dimensional space

data_process.py

    Process the data - takes in the path of the raw data and returns the processed data
    in our case - all with indonesian text - we can change that later if we use multilingual model
    processing realted_text and headline
    some cleaning of the text
    -- make it more detailed later

model
    simcse-embed -- 
        to get the embeddings out of the trained simcse model
        and plot the sentences into 2 dimensional space
         

limitation of the current indosimcse model, and some details:
    SimCSE/output contains the checkpoints -- running run_unsup_example.sh

    trained on top of this model -- indolem/indobertweet-base-uncased
    model based on simcse -- trained via CL on 500k plus tweets
    these tweets are not cleaned to the extent that we actually want them
    this model can be made even better through some training on wiki data (10^6)

eval

    here we fine-tune our model for the task of indoNLI
    works as an evaluation metric if we can outperform the previous models

    we will be conducting experiments on following models:


    indobert-base 
    indobert-lite-base
    indobert-lite-large
    indobert-large 
    https://github.com/IndoNLP/indonlu

    indobertweet-base-uncased (trained on economy, health, education, and government)
    https://github.com/indolem/IndoBERTweet



    and compare them against our simcse model


Preliminary Results: 

    indobert and indosimcse perform pretty much the same --
    might have to train indosimcse on wiki data again 
    
    indoberttweet and indosimcse-tweet are fair to compare but not others
    









