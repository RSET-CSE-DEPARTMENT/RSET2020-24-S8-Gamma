<?php
session_start();
date_default_timezone_set("Asia/Calcutta");
//echo $_SESSION['login_user'];
if($_SESSION['login_user']!="")
//echo "session";
if($_SESSION['login_user']!="admin")
$con=mysqli_connect("localhost","root","","giaan");
else
$con=mysqli_connect("localhost","root","","giaan");
?>