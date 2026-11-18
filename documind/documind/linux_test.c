
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define MAX_NODES 100

struct list_node {
    int data;
    struct list_node *next;
    struct list_node *prev;
};

struct linked_list {
    struct list_node *head;
    struct list_node *tail;
    int size;
};

struct list_node *create_node(int value)
{
    struct list_node *node = malloc(sizeof(struct list_node));

    if (node == NULL)
        return NULL;

    node->data = value;
    node->next = NULL;
    node->prev = NULL;

    return node;
}

void initialize_list(struct linked_list *list)
{
    if (list == NULL)
        return;

    list->head = NULL;
    list->tail = NULL;
    list->size = 0;
}

bool is_empty(struct linked_list *list)
{
    return list == NULL || list->head == NULL;
}

void insert_at_beginning(struct linked_list *list, int value)
{
    struct list_node *node = create_node(value);

    if (node == NULL || list == NULL)
        return;

    node->next = list->head;

    if (list->head != NULL)
        list->head->prev = node;
    else
        list->tail = node;

    list->head = node;
    list->size++;
}

void insert_at_end(struct linked_list *list, int value)
{
    struct list_node *node = create_node(value);

    if (node == NULL || list == NULL)
        return;

    node->prev = list->tail;

    if (list->tail != NULL)
        list->tail->next = node;
    else
        list->head = node;

    list->tail = node;
    list->size++;
}

void delete_node(struct linked_list *list, int value)
{
    if (list == NULL || list->head == NULL)
        return;

    struct list_node *current = list->head;

    while (current != NULL) {
        if (current->data == value) {
            if (current->prev != NULL)
                current->prev->next = current->next;
            else
                list->head = current->next;

            if (current->next != NULL)
                current->next->prev = current->prev;
            else
                list->tail = current->prev;

            free(current);
            list->size--;
            return;
        }

        current = current->next;
    }
}

struct list_node *search_node(struct linked_list *list, int value)
{
    if (list == NULL)
        return NULL;

    struct list_node *current = list->head;

    while (current != NULL) {
        if (current->data == value)
            return current;

        current = current->next;
    }

    return NULL;
}

void reverse_list(struct linked_list *list)
{
    if (list == NULL || list->head == NULL)
        return;

    struct list_node *current = list->head;
    struct list_node *temp = NULL;

    while (current != NULL) {
        temp = current->prev;
        current->prev = current->next;
        current->next = temp;
        current = current->prev;
    }

    if (temp != NULL) {
        list->tail = list->head;
        list->head = temp->prev;
    }
}

void display_list(struct linked_list *list)
{
    if (list == NULL)
        return;

    struct list_node *current = list->head;

    printf("List: ");

    while (current != NULL) {
        printf("%d ", current->data);
        current = current->next;
    }

    printf("\n");
}

void clear_list(struct linked_list *list)
{
    if (list == NULL)
        return;

    struct list_node *current = list->head;

    while (current != NULL) {
        struct list_node *next = current->next;
        free(current);
        current = next;
    }

    list->head = NULL;
    list->tail = NULL;
    list->size = 0;
}

int main(void)
{
    struct linked_list list;

    initialize_list(&list);

    insert_at_end(&list, 10);
    insert_at_end(&list, 20);
    insert_at_beginning(&list, 5);
    insert_at_end(&list, 30);

    display_list(&list);

    struct list_node *found = search_node(&list, 20);

    if (found != NULL)
        printf("Found: %d\n", found->data);

    delete_node(&list, 10);
    display_list(&list);

    reverse_list(&list);
    display_list(&list);

    clear_list(&list);

    return 0;
}
