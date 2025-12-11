-- SELECT * FROM sakila.customer;
-- /* 등록일 create_date이 2006년 2월 14일인 고객을 모두 출력 */
-- SELECT * FROM customer WHERE DATE(create_date) = '2006-02-14';
-- /* customer 테이블에서 이름을 모두 대문자로 출력 */
-- SELECT upper(first_name) AS upper_name FROM customer;
/* customer 테이블에서 이름, 성, 이메일을 -로 구분해 출력 */
SELECT CONCAT(first_name, '-', last_name,'-',email) AS info FROM customer; 
/* customer 테이블에서 이메일 주소에 "sakila"이 포함된 고객 찾으시오 */
 SELECT * FROM customer WHERE email LIKE '%sakila%';