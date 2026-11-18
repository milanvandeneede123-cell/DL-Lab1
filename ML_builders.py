# IMPORTANT NOTE: 
# when you hand in, the two functions in this file must be self-contained, i.e., when we call the functions
# SetLearningParams() and CreateModel() from our own code (without importing anything else you may have put in this file)
# and use them to train a model, this must reproduce your submitted model 

# WARNING: the model architecture and learning parameters used below are just an example of how to do things
#          they are not necessarily good settings


# instantiate all tensorflow elements you use by their full tf name, e.g. tf.keras.layers.Conv2D in order to avoid problems 
# with missing imports in our model evaluation code
# DO NOT add any other import statements here!
import tensorflow as tf

# The function below sets the parameters related to learning convergence (as explained in the first section related to learning convergence in the lecture)
# For each part of the assignment, you are advised to check your learning convergence and change these settings if you think this improves your result

# WARNING: DO NOT CHANGE THE NAME OR INTERFACE OF THIS FUNCTION!!!
def SetLearningParams(): 
    # WARNING: DO NOT CHANGE THE STRUCTURE OF DICT RETURNED BY THIS FUNCTION!!
    
    STUDENT_LASTNAME = "Van den Eede"      #   example:  "Dambre" 
    STUDENT_FIRSTNAME = "Milan"      #   example:  "Joni"
    STUDENT_ID = "02112457"        #   example:  "000386365"

    # make sure student data above is filled in 
    assert STUDENT_ID is not None and STUDENT_LASTNAME is not None and STUDENT_FIRSTNAME is not None, 'Please fill in your First and Last Name and Student Id'
    
    learningparams = {}
    learningparams['student_name'] = f'{STUDENT_LASTNAME} {STUDENT_FIRSTNAME}'
    learningparams['student_seed'] = int(str(STUDENT_ID))

    # set batch size
    learningparams['batch_size'] = 256

    # If you feel you need to use a larger batch size than your system memory allows, you can use gradient accumulation,
    # For, e.g. gradient accumulation accross two steps (=batches), 
    # you effectively double the batch size that determines each weight update step
    # learningparams['gradient_accumulation'] = None
    
    # The number of epochs set below defines the maximal number of epochs in your training run, 
    # This must be long enough for learning to have converged
    # when combined with early stopping, it is advised to set this high enough (at least initially): 
    # this avoids having to restart your training run while the effective number of epochs will be smaller
    
    learningparams['epochs'] = 100
    
    # The line below defines a fixed learning rate, but you can replace it by a learning rate schedule
    # you can find more info on learning rate schedules here:
    # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/LearningRateSchedule
    # you are free to explore and learn, but don't put in too much effort:
    # a good solution without an advanced learning rate schedule (e.g. only reduce-on-plateau) will do for this assignment
    # note that reduce-on-plateau is a callback, so should be specified further in this file
    lr_or_schedule = 5e-4
    learningparams['lrschedule'] = lr_or_schedule
    
    # we advise you to use Adam, but you are free to use alternatives
    learningparams['optimizer'] = tf.keras.optimizers.Adam(lr_or_schedule)

    # Define callbacks: this is where you introduce early stopping when doing long training runs 
    # Note that reduce-on-plateau is also a callback, so should be added here if you want to use it!
    # IT IS ADVISED to keep early stopping in-place (with well-chosen hyperparameter values!!)
    # It combines well with reduce-on-plateau, but you may need to remove it for some other learning rate schedulers
    
    learningparams['callbacks'] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            verbose=1,
            min_delta=1e-3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            verbose=1,
            min_lr=1e-6
        )
    ]
    
    return learningparams



# The function below defines your model architecture, including any architectural elements that improve learning convergence.
# It also allows you to include all regularisation and augmentation options included in this assignment.

# NOTE: in part 2, ideally, you make no more changes to the function below, 
#       compared to the models you handed in in part 1, except for the introduction of dropout layers and regularisation hyperparemeters
#       IF it turns out you really made some poor choices, the only things you are still allowed to change in part 2 are:
#       (1) initial_width, (2) and the final aggregation before the dense layer, i.e. flatten or global pooling
# NOTE: in part 3, you are ONLY allowed to change the regularisation and augmentation in the code below (if desired), compared to your 
#       model submitted in part 2

# WARNING: DO NOT CHANGE THE NAME OR INTERFACE OF THIS FUNCTION!!!
import tensorflow as tf

# WARNING: DO NOT CHANGE THE NAME OR INTERFACE OF THIS FUNCTION!!!
def CreateModel(modelname = '13.Dropout_maxnorm_l2_64_horz', num_classes = 41):

    # change the initial_width below to globally rescale the width of your network 
    initial_width = 64
    maxnorm = tf.keras.constraints.MaxNorm(3.0)
    reg = tf.keras.regularizers.l2(1e-4)
    
    # === INPUTS ===
    inputs = tf.keras.Input(shape=(64, 64, 3))
    scaled = tf.keras.layers.Rescaling(1./255)(inputs)

    # --- DATA AUGMENTATION ---
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
        ],
        name="data_augmentation",
    )
    
    x = data_augmentation(scaled)
    
    # === BLOCK 0 (with skip connection) ===
    conv_0_0 = tf.keras.layers.Conv2D(initial_width, 3, padding='same', kernel_constraint=maxnorm, kernel_regularizer=reg)(x)
    bn_0_0   = tf.keras.layers.BatchNormalization()(conv_0_0)
    act_0_0  = tf.keras.layers.Activation('relu')(bn_0_0)
    
    conv_0_1 = tf.keras.layers.Conv2D(initial_width, 3, padding='same', kernel_constraint=maxnorm, kernel_regularizer=reg)(act_0_0)
    bn_0_1   = tf.keras.layers.BatchNormalization()(conv_0_1)
    
    # --- Add skip connection ---
    proj_0   = tf.keras.layers.Conv2D(initial_width, 1, padding='same', kernel_constraint=maxnorm, kernel_regularizer=reg)(x)  # match channels
    skip_0   = tf.keras.layers.Add()([bn_0_1, proj_0])  # element-wise sum
    act_0_1  = tf.keras.layers.Activation('relu')(skip_0)
    # ---------------------------
    
    red_0  = tf.keras.layers.MaxPooling2D()(act_0_1)
    
    # === BLOCK 1 ===
    conv_1_0 = tf.keras.layers.Conv2D(initial_width * 2, 3, padding='same', kernel_constraint=maxnorm, kernel_regularizer=reg)(red_0)
    bn_1_0   = tf.keras.layers.BatchNormalization()(conv_1_0)
    act_1_0  = tf.keras.layers.Activation('relu')(bn_1_0)
    
    conv_1_1 = tf.keras.layers.Conv2D(initial_width * 2, 3, padding='same', kernel_constraint=maxnorm, kernel_regularizer=reg)(act_1_0)
    bn_1_1   = tf.keras.layers.BatchNormalization()(conv_1_1)
    act_1_1  = tf.keras.layers.Activation('relu')(bn_1_1)
    
    red_1  = tf.keras.layers.MaxPooling2D()(act_1_1)
    
    # === BLOCK 2 ===
    conv_2_0 = tf.keras.layers.Conv2D(initial_width * 4, 3, padding='same', kernel_constraint=maxnorm, kernel_regularizer=reg)(red_1)
    bn_2_0   = tf.keras.layers.BatchNormalization()(conv_2_0)
    act_2_0  = tf.keras.layers.Activation('relu')(bn_2_0)
    
    conv_2_1 = tf.keras.layers.Conv2D(initial_width * 4, 3, padding='same', kernel_constraint=maxnorm, kernel_regularizer=reg)(act_2_0)
    bn_2_1   = tf.keras.layers.BatchNormalization()(conv_2_1)
    act_2_1  = tf.keras.layers.Activation('relu')(bn_2_1)
    
    # Dropout na activatie van blok 2
    drop_2 = tf.keras.layers.SpatialDropout2D(0.1)(act_2_1)
    red_2  = tf.keras.layers.MaxPooling2D()(drop_2)
    
    # === BLOCK 3 ===
    conv_3_0 = tf.keras.layers.Conv2D(initial_width * 8, 3, padding='same', kernel_constraint=maxnorm, kernel_regularizer=reg)(red_2)
    bn_3_0   = tf.keras.layers.BatchNormalization()(conv_3_0)
    act_3_0  = tf.keras.layers.Activation('relu')(bn_3_0)
    
    conv_3_1 = tf.keras.layers.Conv2D(initial_width * 8, 3, padding='same', kernel_constraint=maxnorm, kernel_regularizer=reg)(act_3_0)
    bn_3_1   = tf.keras.layers.BatchNormalization()(conv_3_1)
    act_3_1  = tf.keras.layers.Activation('relu')(bn_3_1)
    
    # Dropout na activatie van blok 3
    drop_3 = tf.keras.layers.SpatialDropout2D(0.2)(act_3_1)
    red_3  = tf.keras.layers.MaxPooling2D()(drop_3)
    
    # === CLASSIFIER ===
    dense_in = tf.keras.layers.GlobalAveragePooling2D()(red_3)
    
    # Dropout op de samengevouwen features (vlak voor Dense)
    drop_head = tf.keras.layers.Dropout(0.5)(dense_in)
    
    logits = tf.keras.layers.Dense(num_classes, kernel_constraint=maxnorm, kernel_regularizer=reg)(drop_head)
    
    conv_model = tf.keras.Model(inputs=inputs, outputs=logits)

    return conv_model