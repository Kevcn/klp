-- Create a new user with admin privileges
CREATE USER admin WITH PASSWORD 'admin_password';

-- Grant superuser privileges to the admin user
ALTER USER admin WITH SUPERUSER;