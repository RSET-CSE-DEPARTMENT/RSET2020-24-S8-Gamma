<html> 
<head><title>Login  Process</title></head>
<body>
<?php
$user_name=$_POST["uname"];
$pass_word=$_POST["passwd"];
$connection = mysqli_connect("localhost","root","");
if(!$connection) 
{
die('Error');
}
$db="db_camera";
$db_selection_status = mysqli_select_db($connection,$db);
if (!$db_selection_status) 
{
die ('Error');
}
$sql_query="select * from tb_users where name='$user_name' and pass='$pass_word' ";
$result=mysqli_query($connection,$sql_query);// Execute SQL query
$num_of_rows= mysqli_num_rows($result); // Processing the Results.
if($num_of_rows==1)
{
session_start();
$_SESSION["username"]=$user_name;
header('Location: live.php');//redirect
}
else
{
echo "Check Username or Password";
}
$connection_close_status=mysqli_close($connection); 
//Close Connection to the MySQL database engine/server.
if(!$connection_close_status)
{
echo "Connection not closed";
}
?>
</body>
<html>

