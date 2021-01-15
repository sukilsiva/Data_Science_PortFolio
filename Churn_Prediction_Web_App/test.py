from pytesting import testing

Data = input("Enter the Data to Be Checked")

test = testing("https://github.com/srivatsan88/YouTubeLI/blob/master/dataset/Wclks")
test.get_data()
test.preprocess_input()
test.testing(Data)


