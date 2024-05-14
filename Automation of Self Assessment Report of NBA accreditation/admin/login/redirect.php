<?php
session_start();
include('../db/connectionI.php');
//$db_name="music"; // Database name 
$tbl_name="employee"; // Table name 

// Connect to server and select databse.


// username and password sent from form 
$myusername=$_POST['UserName']; 
$mypassword=$_POST['Password']; 

// To protect MySQL injection (more detail about MySQL injection)
$myusername = stripslashes($myusername);
$mypassword = stripslashes($mypassword);
$myusername = mysql_real_escape_string($myusername);
$mypassword = mysql_real_escape_string($mypassword);
	
if(isset($_POST['login']))
{
//echo "teacher $myusername $mypassword";

$sql="SELECT * FROM $tbl_name WHERE email='$myusername' and password='$mypassword'";
$result=mysqli_query($con,$sql);

// Mysql_num_row is counting table row
$count=mysqli_num_rows($result);
// If result matched $myusername and $mypassword, table row must be 1 row

if($count==1)
{
// Register $myusername, $mypassword and redirect to file "login_success.php"
 $result = mysqli_query($con,"SELECT * FROM $tbl_name WHERE email='$myusername' and password='$mypassword'") or die('Could not connect: ' . mysqli_error($con));

while($row = mysqli_fetch_array($result))
  {

$_SESSION['company_name'] =$row['company_name'];

if($myusername=="admin")
$_SESSION['user'] ="admin";
else
$_SESSION['user'] ="doctor";
  
$_SESSION['userid'] =$row['id'];
$_SESSION['permission'] =$permission;

date_default_timezone_set("Asia/Calcutta");

$d=date('Y:m:d H:i:s');

$ip=$_SERVER['REMOTE_ADDR'];
$_SESSION['login_user']=$myusername;
//mysql_query("insert into log_in (user_name,date1,ip) values ('$myusername','$d','$ip')") or die ("Error ".mysql_error());
	
	header("location:../dashboard/dashboard.php");
  }
}


else
{

header("location:login.php?a=1");

}





}

?>
 
 



