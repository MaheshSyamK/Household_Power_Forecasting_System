CREATE DATABASE power_consumption;

USE power_consumption;

CREATE USER 'forecast_user'@'localhost' IDENTIFIED BY 'Mahesh@1527';

GRANT ALL PRIVILEGES ON power_consumption.* TO 'forecast_user'@'localhost';

FLUSH PRIVILEGES;

SHOW TABLES;

SELECT COUNT(*) FROM power_consumption;
