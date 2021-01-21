from pytesting import testing

data = input("Enter the DB Address of stored data")
Data = input("Enter the DB Address Data from WEBAPP")

test = testing(original_data=data,webappdata=Data)

print(test.data_testing())


