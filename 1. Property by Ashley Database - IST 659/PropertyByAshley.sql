-- dropping the database if it exists
--DROP SCHEMA IF EXISTS PropertyByAshley

--GO

--creating project database, might have to edit this in the future
--CREATE SCHEMA PropertyByAshley

--GO


--drop Property table before creating it
DROP TABLE IF EXISTS Property

GO
--drop Seller table before creating it
DROP TABLE IF EXISTS Seller

GO


--drop buyer table before creating it
DROP TABLE IF EXISTS Buyer

GO

--drop client table before creating it
DROP TABLE IF EXISTS Client

GO

--Creating the Client table
CREATE TABLE Client(
--Columns for Client Table
Client_id int identity,
Client_First_Name VARCHAR(15) NOT NULL,
Client_Middle_Initial VARCHAR(1) NULL,
Client_Last_Name VARCHAR(15) NOT NULL,
CONSTRAINT PK_Client_ID PRIMARY KEY (Client_id)
)
GO
--End creating the Client Table

--Creating the Buyer Table 
CREATE TABLE Buyer(
--Columns for Buyer Table
Buyer_id int identity,
Buyer_1 int not null,
Buyer_2 int null,
CONSTRAINT PK_Buyer_ID PRIMARY KEY(Buyer_id),
CONSTRAINT FK1_Buyer_1 FOREIGN KEY (Buyer_1) REFERENCES Client(Client_id),
CONSTRAINT FK2_Buyer_2 FOREIGN KEY (Buyer_2) REFERENCES Client(Client_id)
)
GO
--End creating Buyer Table


--Creating Seller Table
CREATE TABLE Seller(
--Columns for Seller Table
Seller_id int identity,
Seller_1 int NOT NULL,
Seller_2 int NULL,
CONSTRAINT PK_Seller_ID PRIMARY KEY(Seller_id),
CONSTRAINT FK1_Seller_1 FOREIGN KEY (Seller_1) REFERENCES Client(Client_id),
CONSTRAINT FK2_Seller_2 FOREIGN KEY (Seller_2) REFERENCES Client(Client_id)
)
GO
--End creating Seller Table


--Creating Property Table
CREATE TABLE Property(
--Column for Property Table
Property_id int identity,
Property_Address VARCHAR(50) NOT NULL,
Property_City VARCHAR(15) NOT NULL,
Property_Zipcode VARCHAR(5) NOT NULL,
Property_Sell_Price DECIMAL(30,2) NOT NULL,
Property_Commission_Percentage DECIMAL(3,2) NULL,
Property_To_Market_Date DATETIME NOT NULL,
Property_Sold_Date DATETIME NOT NULL,
Buyer_id int NULL,
Seller_id int NULL,
CONSTRAINT PK_Property_ID PRIMARY KEY (Property_id),
CONSTRAINT FK1_Buyers FOREIGN KEY (Buyer_id) REFERENCES Buyer(Buyer_id),
CONSTRAINT FK2_Sellers FOREIGN KEY (Seller_id) REFERENCES Seller(Seller_id)
)
GO
--End creating Property Table





--Adding data to Client Table

INSERT INTO dbo.Client (Client_First_Name, Client_Last_Name)
	VALUES 
	('Joseph', 'Momich'),
	('Michelle', 'Momich'),
	('Khristian', 'Avelar'),
	('Larry', 'Paskow'),
	('Inguss', 'Strikaitis'),
	('Gabriel', 'Williams'),
	('Stephanie', 'Banh'),
	('Edwin', 'Lew'),
	('Janelle', 'Bode'),
	('Lee', 'Austria'),
	('Amber', 'McCurry'),
	('Beverly', 'Calpito'),
	('Scott', 'Saling'),
	('Lora', 'Graham'),
	('Ann', 'Berg'), 
	('Lilian','Lew'), 
	('Josephine', 'Cheung'),
	('Bob', 'Tang')

	--End adding data to Client Table
	
	--Adding data to Buyer Table
	INSERT INTO Buyer (Buyer_1, Buyer_2) VALUES 
	(1,2), (4,5), (6,7), (10,11), (12,13)

	INSERT INTO Buyer (Buyer_1) VALUES (8),(15),(16),(17),(16),(18)

	--End adding data to Buyer Table


	--Adding data to Seller Table
	INSERT INTO Seller (Seller_1) VALUES (3), (9), (14), (16)

	


	--Adding data to Property Table
	--Inserting properties with sellers represented in the transaction
	INSERT INTO Property (Property_Address, Property_City, Property_Zipcode, Property_Sell_Price,
	Property_To_Market_Date, Property_Sold_Date, Seller_id) VALUES 
	('272 Magda Way', 'Pachecho', '94553', 150000.00, '9/9/2021', '9/29/2021', 2),
	('1404 Henry Street', 'Berkeley', '94709', 668000.00, '6/17/2021', '8/18/2021', 3),
	('2925 Grande Corte', 'Walnut Creek', '94598', 1225000.00, '09/08/2020', '10/13/2020', 4)



	--Inserting properties with buyers and sellers represented in single transaction
	INSERT INTO Property (Property_Address, Property_City, Property_Zipcode, Property_Sell_Price,
	Property_To_Market_Date, Property_Sold_Date, Buyer_id, Seller_id) VALUES 
	('1505 Kirker Pass Road, #161', 'Concord', '94521', 325000.00, '12/01/2021', '12/22/2021', 1,1)

	--Inserting properties where the buyers was reperesented in transcations
	INSERT INTO Property (Property_Address, Property_City, Property_Zipcode, Property_Sell_Price,
	Property_To_Market_Date, Property_Sold_Date, Buyer_id) VALUES 
	('1255 Detroit Avenue, #22', 'Concord', '94520', 470000.00, '11/29/2021', '12/22/2021', 2),
	('378 Topaz Street', 'Brentwood', '94513', 740888.00, '11/02/2021', '12/07/2021', 3),
	('4710 Colorado Court', 'Camino', '95709', 615000.00, '09/17/2021', '10/26/2021', 6),
	('840 Flores Way', 'Rio Vista', '94571', 470000.00, '08/05/2021', '09/02/2021', 4),
	('1312 Tuolumne Way', 'Oakley', '94561', 756000.00, '07/22/2021', '08/20/2021', 5),
	('5212 Clovis Court', 'Concord', '94521', 945000.00, '06/01/2021', '07/09/2021', 7),
	('1325 Rimer Drive', 'Moraga', '94556', 1041000.00, '11/30/2020', '12/18/2020', 10),
	('3191 Tiffanie Lane', 'Napa', '94558', 1200000.00, '09/22/2020', '11/16/2020', 9),
	('2925 Grande Corte', 'Walnut Creek', '94598', 910000.00, '03/04/2020', '03/05/2020', 10),
	('50 Rainbow Circle', 'Danville', '94506', 885000.00, '09/18/2019', '11/22/2019', 11)




	--View all clients
SELECT * FROM Client

--View all Properties
SELECT Property.Property_Address, Property.Property_City, 
Property.Property_Sell_Price FROM Property

--View all Buyers
SELECT Buyer.Buyer_id, Client.Client_First_Name, Client.Client_Last_Name FROM Buyer JOIN Client
	on Buyer.Buyer_1 = Client.Client_id OR Buyer.Buyer_2 = Client.Client_id
--Results:


--View all Sellers
SELECT Seller.Seller_id, Client.Client_First_Name, Client.Client_Last_Name FROM Seller JOIN Client
	on Seller.Seller_1 = Client.Client_id OR Seller.Seller_2 = Client.Client_id
GO


--Creating a View to see All Buyers
CREATE OR ALTER VIEW Buyers AS
	SELECT Buyer.Buyer_id, Client.Client_First_Name, Client.Client_Last_Name FROM Buyer JOIN Client
	on Buyer.Buyer_1 = Client.Client_id OR Buyer.Buyer_2 = Client.Client_id
GO

SELECT * FROM Buyers
GO
--Creating a View to see All Sellers

CREATE OR ALTER VIEW Sellers AS
	SELECT Seller.Seller_id, Client.Client_First_Name, Client.Client_Last_Name FROM Seller JOIN Client
	on Seller.Seller_1 = Client.Client_id OR Seller.Seller_2 = Client.Client_id
GO
SELECT * FROM Sellers
GO
	
SELECT * FROM Property

SELECT Property.Property_Address, Property.Property_City, Property.Property_Zipcode, Property.Property_Sell_Price, Buyer.Buyer_1, 
Buyer.Buyer_2, Seller.Seller_1, Seller.Seller_2 FROM Property LEFT JOIN Buyer ON Property.Buyer_id = Buyer.Buyer_id 
LEFT JOIN Seller ON Property.Seller_id = Seller.Seller_id
GO



--Creating a View to see All Buyers and Property Details

CREATE OR ALTER VIEW PropertyBuyers AS
	SELECT Property.Property_Address, Property.Property_City, Property.Property_Zipcode, Property.Property_Sell_Price, Client.Client_First_Name, Client.Client_Last_Name FROM Property
	LEFT JOIN Buyer ON Property.Buyer_id = Buyer.Buyer_id JOIN Client ON Client.Client_id = Buyer.Buyer_1 OR Client.Client_id = Buyer.Buyer_2
GO
SELECT * FROM PropertyBuyers
GO
--Results:


--Creating a View to see All Sellers and Property Details

CREATE OR ALTER VIEW PropertySellers AS
		SELECT Property.Property_Address, Property.Property_City, Property.Property_Zipcode, Property.Property_Sell_Price, Client.Client_First_Name, Client.Client_Last_Name FROM Property
		LEFT JOIN Seller ON Property.Seller_id = Seller.Seller_id JOIN Client ON Client.Client_id = Seller.Seller_1 OR Client.Client_id = Seller.Seller_2
GO
SELECT * FROM PropertySellers
--Results:




--What is the Minimum, Average, and Maximum Home Sales Price?
SELECT MIN(Property.Property_Sell_Price) AS MinSalesPrice, 
AVG(Property.Property_Sell_Price) AS AvgSalesPrice,
MAX(Property.Property_Sell_Price) AS MaxSalesPrice
FROM Property
--Results:

--What is the Average Home Sales Price for each Month?
SELECT MONTH(Property.Property_Sold_Date) AS SellMonth, YEAR(Property.Property_Sold_Date) AS SellYear, 
AVG(Property.Property_Sell_Price) AS AvgSellPrice
FROM Property
GROUP BY MONTH(Property.Property_Sold_Date), YEAR(Property.Property_Sold_Date)	
ORDER BY AvgSellPrice DESC
GO
--Results:
SELECT * FROM Property

--What are the Average Number of days on the market per year?

SELECT YEAR(Property.Property_Sold_Date) AS YearofTransaction,
AVG(DATEDIFF(DAY, Property.Property_To_Market_Date, Property.Property_Sold_Date)) AS AvgDaysOnMarket 
FROM Property GROUP BY YEAR(Property.Property_Sold_Date)

--What are the Average Number of days on the market per city?
SELECT Property.Property_City AS City,
AVG(DATEDIFF(DAY, Property.Property_To_Market_Date, Property.Property_Sold_Date)) AS AvgDaysOnMarket 
FROM Property GROUP BY Property.Property_City
--Results:

--What are the Average Number of days on the market for each month?

SELECT MONTH(Property.Property_Sold_Date) AS MonthofTransaction,
AVG(DATEDIFF(DAY, Property.Property_To_Market_Date, Property.Property_Sold_Date)) AS AvgDaysOnMarket 
FROM Property GROUP BY MONTH(Property.Property_Sold_Date)
GO
--Results:
	
--Creating a procedure to add a new Client 
CREATE OR ALTER PROCEDURE NewClient (@firstName varchar(30), @middleInitial varchar(30), @lastName varchar(30))
AS
BEGIN
	INSERT INTO Client (Client.Client_First_Name, Client.Client_Middle_Initial, Client.Client_Last_Name)
	VALUES (@firstName, @middleInitial, @lastName)
END
GO

SELECT * FROM Client
EXEC NewClient 'Jazmin', 'M', 'Logrono' 
SELECT * FROM Client
--DELETE  Client WHERE Client.Client_First_Name = 'Jazmin'
GO

--Creating a procedure for adding a Client to the Seller table
CREATE OR ALTER PROCEDURE NewSeller (@clientID int)
AS 
BEGIN
	INSERT INTO Seller (Seller.Seller_1) VALUES (@clientID)
END
GO
EXEC NewSeller 19
GO

--Creating a procedure for adding a Client to the Buyer table
CREATE OR ALTER PROCEDURE NewBuyer (@clientID int)
AS 
BEGIN
	INSERT INTO Buyer (Buyer.Buyer_1) VALUES (@clientID)
END
GO

EXEC NewBuyer 19

GO

--Creating a procedure for adding a Property to the Property table
CREATE OR ALTER PROCEDURE NewProperty (@propertyAddress varchar(30), @propertyCity varchar(30), 
		@propertyZipcode varchar(5), @propertySellPrice DECIMAL(30,2), @propertyToMarket DATETIME, 
		@propertySoldDate DATETIME)
AS 
BEGIN
 INSERT INTO Property (Property.Property_Address, Property.Property_City, Property.Property_Zipcode,
 Property.Property_Sell_Price, Property.Property_To_Market_Date, Property.Property_Sold_Date) 
 VALUES (@propertyAddress, @propertyCity, @propertyZipcode, @propertySellPrice, 
 @propertyToMarket, @propertySoldDate)
END
GO

EXEC NewProperty '320 N Civic Drive', 'Walnut Creek', 94596, 260000.00, '05/12/2018', '6/28/2018'


SELECT * FROM Seller
SELECT * FROM Buyer
SELECT * FROM Client
SELECT * FROM Property
GO



--Defining user for stakeholder
--CREATE USER ashlew FOR LOGIN ashlew
--GRANT EXECUTE ON NewClient TO ashlew
--GRANT EXECUTE ON NewBuyer TO ashlew
--GRANT EXECUTE ON NewSeller TO ashlew
--GRANT SELECT ON PropertyBuyers TO ashlew
--GRANT SELECT ON PropertySellers TO ashlew 


