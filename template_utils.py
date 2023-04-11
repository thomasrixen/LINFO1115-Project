#if needed, write here your additional fuctions/classes with their signature and use them in the exercices:
# a specific place is available to copy them at the end of the Inginious task.

#First, import the libraries needed for your helper functions
import numpy as np
import pandas as pd
#Then write the classes and/or functions you wishes to use in the exercises
def example_helper_function(arg1, arg2):
    return 0

class Student:
    def __init__(self, id, dataframe):
        self.id = id
        self.messageReceived = dataframe.loc[dataframe["Src"] == id, "Dst"]
        self.messageSent = dataframe.loc[dataframe["Dst"] == id, "Src"]
        self.sentMessageContact = self.messageSent.drop_duplicates()
        self.receivedMessageContact = self.messageReceived.drop_duplicates()
        self.contact = pd.concat([self.sentMessageContact, self.receivedMessageContact]).drop_duplicates()
    
    def nbrOfMessagesSentTo(self, otherStudentId):
        """
            input: otherStudentId = the id of a other student
            output: the number of message that the student sent to the otherStudentId
        """
        return self.getMessageSent().loc[self.getMessageSent() == otherStudentId].size

    def nbrOfMessagesReceivedFrom(self, otherStudentId):
        """
            input: otherStudentId = the id of a other student
            output: the number of message that the student received from the otherStudentId
        """
        return self.getMessageReceived().loc[self.getMessageReceived() == otherStudentId].size

    def getId(self):
        return self.id
    
    def getMessageReceived(self):
        return self.messageReceived
    
    def getMessageSent(self):
        return self.messageSent
    
    def getSentMessageContact(self):
        return self.sentTo
    
    def getReceivedMessageContact(self):
        return self.receivedFrom
    
    def getContact(self):
        return self.contact
