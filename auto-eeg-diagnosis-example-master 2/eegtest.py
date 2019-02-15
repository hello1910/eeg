from keras_generator_final import MygeneratorVal  #startshere!
from dataset import get_all_sorted_file_names_and_labels
from keras.models import load_model
from igloo1d import PATCHY_LAYER_RETURNFULLSEQ
from kerasgetmodel import get_model
from sklearn.externals import joblib
import numpy as np
import nltk

def test_file():

    all_file_names,irrelevantlabels=get_all_sorted_file_names_and_labels('eval',['../normal/',
        '../abnormal/'])
    model=get_model()
    model.load_weights('eegv2.h5')

    word_to_token=joblib.load('words_to_tokens')
    token_to_word = dict([[v,k] for k,v in word_to_token.items()])

    def tok2word(tok_seq):
        caption=' '
        for i in seq_out[0][0]:
            caption+=token_to_word[int(i)]+ ' '
        return caption

    def generate_caption(x_first,modely,max_tokens=9):
        """
        Generate a caption for the image in the given path.
        The caption is limited to the given number of tokens (words).
        """

        # Load and resize the image.
       
        
        # Expand the 3-dim numpy array to 4-dim
        # because the image-model expects a whole batch as input,
        # so we give it a batch with just one image.
        #image_batch = np.expand_dims(image, axis=0)

        # Process the image with the pre-trained image-model
        # to get the transfer-values.
        #transfer_values = image_model_transfer.predict(image_batch)

        # Pre-allocate the 2-dim array used as input to the decoder.
        # This holds just a single sequence of integer-tokens,
        # but the decoder-model expects a batch of sequences.
        shape = (1, max_tokens)
        decoder_input_data = np.zeros(shape=shape, dtype=np.int)

        # The first input-token is the special start-token for 'ssss '.
        token_int = 1

        # Initialize an empty output-text.
        output_text = ''

        # Initialize the number of tokens we have processed.
        count_tokens = 1

        # While we haven't sampled the special end-token for ' eeee'
        # and we haven't processed the max number of tokens.
        while token_int != 'eeee' and count_tokens < max_tokens:
            # Update the input-sequence to the decoder
            # with the last token that was sampled.
            # In the first iteration this will set the
            # first element to the start-token.
            decoder_input_data[0, count_tokens] = token_int

            # Wrap the input-data in a dict for clarity and safety,
            # so we are sure we input the data in the right order.
            x_data = \
            {
                'input': x_first,
                'decoder_input': decoder_input_data
            }

            # Note that we input the entire sequence of tokens
            # to the decoder. This wastes a lot of computation
            # because we are only interested in the last input
            # and output. We could modify the code to return
            # the GRU-states when calling predict() and then
            # feeding these GRU-states as well the next time
            # we call predict(), but it would make the code
            # much more complicated.
            
            # Input this data to the decoder and get the predicted output.
            irrelevant,decoder_output = modely.predict(x_data)

            # Get the last predicted token as a one-hot encoded array.
            # Note that this is not limited by softmax, but we just
            # need the index of the largest element so it doesn't matter.
            token_onehot = decoder_output[0, count_tokens, :]

            # Convert to an integer-token.
            token_int = np.argmax(token_onehot)

            # Lookup the word corresponding to this integer-token.
            sampled_word = token_to_word[token_int]

            # Append the word to the output-text.
            output_text += " " + sampled_word

            # Increment the token-counter.
            count_tokens += 1

        # This is the sequence of tokens output by the decoder.
        output_tokens = decoder_input_data[0]

        # Plot the image.
        #plt.imshow(image)
        #plt.show()
        
        return output_text

    gen_val=MygeneratorVal()
    Acc=0
    TP=0
    FP=0
    TN=0
    FP=0

    neg_count=0

    BLEU_score=0

    for i in range(276):
        try:
            [x_in,seq_in],[x_bin,seq_out]=gen_val[2]
            ans=model.predict([x_in,seq_in],batch_size=1)
        
            if int(ans[0][0][0]) == int(x_bin[0][0]):
                Acc+=1
                if int(ans[0][0][0])==1:
                    TP+=1
                else:
                    TN+=1
            else:
                if int(ans[0][0][0])==1:
                    FP+=1
                else:
                    FN+=1
            caption=generate_caption(x_in,model)
            
            right_string=tok2word(seq_out)
            
            BLEUscore = nltk.translate.bleu_score.corpus_bleu(ans.split(" "),ans2.split(" "))
            BLEU_score+=BLEUscore
        except:
            neg_count+=1
            
    denom=276-neg_count
    file = open('binresults.txt','w') 
    file.write(str(Acc/denom)+" "+"TP"+ str(TP)+" "+ "TN" + str(TN)+" "+"FP"+ str(FP)+" "+"FN"+ str(FN))
    file.write("bleu"+ str(BLEUscore))
    file.close()
