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
