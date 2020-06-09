# Robotic-Process-Automation-Email-Query-processing
Automated email-based reply to customer feedback using sentimental analysis and robotic process automation.

https://github.com/Spurthisk/Robotic-Process-Automation-Email-Query-processing/blob/master/system%20model.PNG


# Description of proposed system
	Customers send the feedback to our system through mail.
	Our system classifies the customers reviews into positive and negative feedback with the help of sentimental analysis.
	Based on the customers reviews replies are sent through the emails i.e “Thank you” mail for a positive feedback and for negative feedback reply is sent by describing the failure of the service.

# Proposed Methodology
The software used to develop Automated Email Reply Processing is Robotic Process Automation tool i.e. UiPath studio along with excel sheet to store the data extracted from the emails sent by the customers. The methodology used is Waterfall Model. Waterfall Model is one of the software development life cycle model. Users proceed to next phase if and only if current phase is complete. Firstly, team collected requirements and analyzed it. Construction phase is important in waterfall model and very time consuming. To read and store the review emails sent by customers we used UiPath which reads and stores Body, subject along with sender email address to excel sheet. Another process in UiPath reads the data collected in excel sheet and apply sentimental analysis to the reviews and categorizes the reviews as positive and negative reviews. Based on which sender will receive the reply email. For testing module, it is categorized into unit testing, acceptance testing and integration testing. Once the bugs are found they are fixed before integrating whole system. Once the bot is developed, maintenance is mandatory to ensure the proper working of the project.

# Sentimental Analysis
