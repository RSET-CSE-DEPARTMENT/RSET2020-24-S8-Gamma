<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>User Addition Status</title>
<style>
    body {
        font-family: Arial, sans-serif;
        background-color: gold;
        margin: 0;
        padding: 0;
    }

    h1 {
        text-align: center;
        color: #333;
    }

    .container {
        max-width: 400px;
        margin: 50px auto;
        padding: 20px;
        background-color: #fff;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        text-align: center;
    }

    .success {
        color: green;
    }

    .error {
        color: red;
    }
</style>
</head>
<body>
<div class="container">
    <h1>User Addition Status</h1>
    <?php
    $name = $_GET["name"];
    $pass = $_GET["pass"];
    $connection = mysqli_connect("localhost", "root", ""); // Establish Connection to MySQL database Engine/server
    if (!$connection) {
        die('Connection to MySQL database engine/server Failed : ' . mysqli_connect_error());
    }
    $db = "db_camera";
    $db_selection_status = mysqli_select_db($connection, $db); // Establish Connection to a specific database.

    if (!$db_selection_status) {
        die ("Can\'t establish connection to the Data Base $db : " . mysqli_error());
    }

    $sql_query = "insert into tb_users (name, pass) values ('$name','$pass')";
    $status = mysqli_query($connection, $sql_query); // Execute SQL query

    if (!$status) {
        echo "<p class='error'>User Addition Failed</p>";
    } else {
        echo "<p class='success'>User Added Successfully</p>";
    }

    $connection_close_status = mysqli_close($connection); // Close Connection to the MySQL database engine. 
    if (!$connection_close_status) {
        echo "<p class='error'>Connection to MySQL database engine is not closed</p>";
    }
    ?>
</div>
</body>
</html>
