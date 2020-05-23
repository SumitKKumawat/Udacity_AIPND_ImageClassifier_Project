# Use a Pre-trained Image Classifier to Identify Dog Breeds

## Image Classification for a City Dog Show

#### Project Goal
* Improving your programming skills using `Python`


In this project you will use a created image classifier to identify dog breeds. We ask you to focus on Python and not on the actual classifier (We will focus on building a classifier ourselves later in the program).


### Description:

Your city is hosting a citywide dog show and you have volunteered to help the organizing committee with contestant registration. Every participant that registers must submit an image of their dog along with biographical information about their dog. The registration system tags the images based upon the biographical information.


Some people are planning on registering pets that **aren’t actual dogs**.


You need to use an already developed `Python` classifier to make sure the participants are dogs.


**Note, you DO NOT need to create the classifier. It will be provided to you. You will need to apply the Python tools you just learned to USE the classifier.**

### Your Tasks:
* Using your `Python` *skills*, you will determine which image classification algorithm works the **"best"** on classifying images as "dogs" or "**not** dogs".

* Determine how well the "**best**" classification algorithm works on correctly identifying a dog's breed.
If you are confused by the term image classifier look at it simply as a tool that has an input and an output. The Input is an image. The output determines what the image depicts. (for example: a dog). Be mindful of the fact that image classifiers do not always categorize the images correctly. (We will get to all those details much later on the program).

* Time how long each algorithm takes to solve the classification problem. With computational tasks, there is often a trade-off between accuracy and runtime. The more accurate an algorithm, the higher the likelihood that it will take more time to run and use more computational resources to run.

For further clarifications, please check our [FAQs](https://github.com/udacity/AIPND-revision/blob/master/notes/project_intro-to-python.md) here.

### Important Notes:
For this image classification task you will be using an image classification application using a deep learning model called a convolutional neural network (often abbreviated as CNN). CNNs work particularly well for detecting features in images like colors, textures, and edges; then using these features to identify objects in the images. You'll use a CNN that has already learned the features from a giant dataset of 1.2 million images called [ImageNet](http://www.image-net.org). There are different types of CNNs that have different structures (architectures) that work better or worse depending on your criteria. With this project you'll explore the three different architectures (**AlexNet**, **VGG**, and **ResNet**) and determine which is best for your application.

We have provided you with a *classifier function* in `classifier.py` that will allow you to use these CNNs to classify your images. The `test_classifier.py` file contains an example program that demonstrates how to use the *classifier function*. For this project, you will be focusing on using your Python skills to complete these tasks using the *classifier function*; in the Neural Networks lesson you will be learning more about how these algorithms work.

Remember that certain breeds of dog look very similar. The more images of two similar looking dog breeds that the algorithm has learned from, the more likely the algorithm will be able to distinguish between those two breeds. We have found the following breeds to look very similar: [Great Pyrenees](https://www.google.com/search?q=Great+Pyrenees&source=lnms&tbm=isch&sa=X&ved=0ahUKEwje252-kpfZAhVF3FMKHeXwB3IQ_AUICigB&biw=1112&bih=1069) and [Kuvasz](https://www.google.com/search?tbm=isch&q=Kuvasz&spell=1&sa=X&ved=0ahUKEwi9_9fTkpfZAhWB7FMKHXlKDWoQBQg6KAA&biw=1112&bih=1069&dpr=1), [German Shepherd](https://www.google.com/search?biw=1112&bih=1069&tbm=isch&sa=1&ei=d7F8WpaaMc_VzgLW8LvABw&q=German+Shepherd&oq=German+Shepherd&gs_l=psy-ab.3..0i67k1j0l2j0i67k1j0l6.31751.41069.0.41515.29.18.4.7.9.0.131.1164.14j2.17.0....0...1c.1.64.psy-ab..2.26.1140.0..0i10k1j0i13k1.112.xUB8_AoVF9w) and [Malinois](https://www.google.com/search?biw=1112&bih=1069&tbm=isch&sa=1&ei=orF8WtHWDcOdzwLnyLXgBw&q=Malinois&oq=Malinois&gs_l=psy-ab.3..0l3j0i67k1l3j0l2j0i67k1j0.31864.42125.0.42493.23.20.0.1.1.0.132.1460.14j4.19.0....0...1c.1.64.psy-ab..8.14.926.0...75.U5aOu6JZ9Vk), [Beagle](https://www.google.com/search?biw=1112&bih=1069&tbm=isch&sa=1&ei=zbF8WqTiHZDxzgKlm5SYBw&q=Beagle&oq=Beagle&gs_l=psy-ab.3..0i67k1j0l2j0i67k1l2j0l5.29396.33482.0.34041.12.8.3.1.1.0.126.585.6j2.8.0....0...1c.1.64.psy-ab..0.12.609...0i10k1.0.Dr92CW2Kqqo) and [Walker Hound](https://www.google.com/search?biw=1112&bih=1069&tbm=isch&sa=1&ei=8LF8WteAGND0zgKvlL-IBw&q=Walker+hound&oq=Walker+hound&gs_l=psy-ab.3..0l10.20697.23454.0.23773.12.10.0.2.2.0.81.601.10.10.0....0...1c.1.64.psy-ab..0.12.610...0i67k1.0.GI0QxI1sadY), amongst others.



## Project Instructions
### Principal Objectives
1. Correctly identify which pet images are of dogs (even if breed is misclassified) and which pet images aren't of dogs.
 
2. Correctly classify the breed of dog, for the images that are of dogs.
 
3. Determine which CNN model architecture (ResNet, AlexNet, or VGG), "best" achieve the objectives 1 and 2.
 
4. Consider the time resources required to best achieve objectives 1 and 2, and determine if an alternative solution would have given a "good enough" result, given the amount of time each of the algorithms take to run.


## TODO:

### Edit program `check_images.py`
The `check_images.py` is the program file that you will be editing to achieve the four objectives above. This file contains a `main()` function that outlines how to complete this program through using functions that have not yet been defined. You will be creating these undefined functions in `check_images.py` to achieve the objectives above.

All of the `TODOs` are listed in `check_images.py`. You will find further elaborations and explanations for each, in the following concepts of this project.


**If you feel that you need more guidance, please refer to the files ending with`_hints.py`. In the workspace you will find a hint file for each of the tasks.**

### Important notes:
* Before beginning the project please review the [Frequently Asked Questions](https://github.com/udacity/AIPND-revision/blob/master/notes/project_intro-to-python.md), FAQ, about the project.

* This project and other lessons within the Nanodegree will be using a [GitHub repository](https://github.com/udacity/AIPND-revision) to store program files and other resources for this Nanodegree. To learn more about **GitHub**, please see the **GitHub** Lesson that's located within the **Extracurricular** (optional) section of this Nanodegree.

* The **Project Workspace** is set up with the programs and files (like pet_images folder) you will need to complete the project.

* The Python comments that begin with `# TODO:` in the `check_images.py` program indicates where you will need to change the code of the program. The comments in `check_images.py` will help you make the changes needed.

* Function docstrings contain input parameters and return values, which were left to provide guidance. You are welcome to program these functions differently.

* In 6. Timing Code to 19. Printing Results we will provide additional guidance for programming the undefined functions and completing the check_images.py program. This information has been provided to help you through the process.


The information provides:

    * Which Lessons to review regarding programming the undefined functions.
    * Details about the assignment's files (e.g. image files in pet_images folder, dognames.txt).
    * Details regarding using the classifier function in classifier.py.
    * Links to relevant python documentation.
    * Relevant example code.

* You can use the functions within the program [print_functions_for_lab_checks.py](https://github.com/udacity/AIPND-revision/blob/master/intropyproject-classify-pet-images/print_functions_for_lab_checks.py) to check your code for sections **8. Command Line Arguments** through **17. Calculating Results**. You will find this program within the Project Workspace and within the [GitHub repository](https://github.com/udacity/AIPND-revision).

### Program Outline
* Time your program
  * Use Time Module to compute program runtime
  
* Get program Inputs from the user
  * Use command line arguments to get user inputs
  
* Create Pet Images Labels
  * Use the pet images filenames to create labels
  * Store the pet image labels in a data structure (e.g. dictionary)
  
* Create Classifier Labels and Compare Labels
  * Use the Classifier function to classify the images and create the classifier labels
  * Compare Classifier Labels to Pet Image Labels
  * Store Pet Labels, Classifier Labels, and their comparison in a complex data structure (e.g. dictionary of lists)
  
* Classifying Labels as "Dogs" or "Not Dogs"
  * Classify all Labels as "Dogs" or "Not Dogs" using dognames.txt file
  * Store new classifications in the complex data structure (e.g. dictionary of lists)
  
* Calculate the Results
  * Use Labels and their classifications to determine how well the algorithm worked on classifying images
  
* Print the Results


You will need to repeat these tasks for each of the three image classification algorithms that are provided to you.




## `#TODO:` 1: Command Line Arguments

**Fill code in the get_input_args() function to create & retrieve the command line arguments**

### Code to Edit
This section will help you code the function **get_input_args** within **get_input_args.py**. With this function you will use argparse to retrieve three command line arguments from the user. (Argparse makes it easy to write user-friendly command-line interfaces).

* Code for the function definition `def get_input_args():` as indicated by `#TODO:` 1within **get_input_args.py**.

### Expected Outcome
When completed this code will input the three command line arguments from the user.

### Checking your code
The **check_command_line_arguments** function within `check_images.py` will check your code.

Test the following:

* Entering no command line arguments when you run `check_image.py` from the terminal window. This should result in the default values being printed.
* Entering in values of your choosing for the command line arguments when you run `check_image.py` from the terminal window. This should result in the values you entered being printed.


#### Project Workspace - Command Line Arguments
* The next concept will have your workspace to work on `#TODO: 1`
* Editing of **check_image.py** and **get_input_args.py** can be done within the **Project Workspace - Command Line Arguments**


### For additional information and help on `#TODO: 1`, please look at the information below:
#### Purpose
The purpose of command line arguments is to provide a way for your programs to be more flexible by allowing external inputs (command line arguments) to be input into a program. The key is that these external arguments can change as to allow more flexibility in the program.

For example, imagine you wrote a program that simply counts the number of lines in a file and prints out that number to the screen. To allow the user to enter in any file without having to change the program, one would want to pass in the file location as a command line argument. In this way, the program could be used on any file since the value is passed as an external input at runtime.

#### Usage of Argparse:
We will be using the **argparse** module to input the following external inputs into our program **check_image.py**. We recommend writing the **get_input_args function** to get the command line arguments using argparse.

Below are the three external inputs your **check_image.py** program will need to retrieve from the user along with the suggested default values each should have.

* Folder that contains the pet images
  * pet_images/
  
* The CNN model architecture to use
  * resnet, alexnet, or vgg (pick one as the default). You will find them in classifier.py.

* The file that contains the list of valid dognames
  * dognames.txt
 

The **get_input_args** function will need to create an argument parser object using [argparse.ArgumentParser](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser) and then use the [add_argument method](https://docs.python.org/3/library/argparse.html#adding-arguments) to allow the users to enter in these three external inputs from above.

Below is an example of creating an argument parser object and then using **add_argument** to add an argument that's a path to a folder and a second argument that's an integer.

```python
# Creates Argument Parser object named parser
parser = argparse.ArgumentParser()

# Argument 1: that's a path to a folder
parser.add_argument('--dir', type = str, default = 'pet_images/', 
                    help = 'path to the folder of pet images')
```

Below you will find an explanation of the inputs into **add_argument**.

* **Argument 1:**

  * --dir = The variable name of the argument (here it's dir)

  * type = The type of the argument (here it's a string)

  * default = The default value (here it's 'pet_images/')

  * help = The text that will appear if the user types the program name and then -h or --help. This allows the user to understand the what's expected an argument's value

### Accessing Argparse Arguments
To access the arguments passed into the program through your argparse object, you will need to use the [parse_args method](https://docs.python.org/3/library/argparse.html#the-parse-args-method). The code below demonstrates how to access the arguments through the argparse extending the example above.

To begin, you will need to assign a variable to **parse_args** and then use that variable to access the arguments of your argparse object. If you are creating the argparse object within a function, you will need to return **parse_args** instead of assigning a variable to it. Also note that the variable **in_args** points to a collection of the command line arguments.

This means to access the one we created in the code above, we have to reference the collection variable name **in_args** then specify the command line argument variable name `dir`. For this example, it would be **in_args.dir**, where in_args is the collection variable name and dir refers to the command line argument variable name. Notice that you need a dot (.) separating the two variable names. The code below shows the assignment of **in_args** to our parser and then accessing the value of **in_args.dir** with the print statement.

```python
# Assigns variable in_args to parse_args()
in_args = parser.parse_args()

# Accesses values of Argument 1 by printing it
print("Argument 1:", in_args.dir)
```

### Running a Program using command line arguments
To run a program like **check_images.py**, first open a terminal window within the Project Workspace. Next type the following and hit enter to run the program (this example - check_images.py). Because no command line arguments are specified after the program name (this example - check_images.py) this will use the default command line arguments that have been defined.

`python check_images.py`

To run a program like **check_images.py** using the command line argument `--dir`, first open a terminal window within the Project Workspace. Next type the following and hit enter to run the program (this example - check_images.py). Notice that all command line arguments are specified after the program name (this example - check_images.py) and they are indicated by the -- that proceeds their variable name (this example : dir) with the value following the variable name (in this example the string : pet_images/).

`python check_images.py --dir pet_images/`

If you are having difficulty running **check_images.py** with command line arguments, see the example program call on line 23 of the program.

